import copy
import torch
import inspect
import warnings
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Callable
import torch.distributed as dist
import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils._sklearn_available = False

from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationConfig,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput, 
    GenerationMixin,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    is_deepspeed_zero3_enabled,
    is_fsdp_managed_module,
    # is_torchdynamo_compiling,
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    is_hqq_available,
    QuantizedCacheConfig,
    # is_quanto_available,
    DynamicCache,
    EncoderDecoderCache,
    logging
)
# from transformers.generation.stopping_criteria import validate_stopping_criteria

logger = logging.get_logger(__name__)


class GenerationWithCTC(GenerationMixin):

    # position_ids=position_ids,
    # attention_mask=attention_mask,
    # inputs_embeds=inputs_embeds,
    # output_hidden_states=True,
    # return_dict_in_generate=True,
    # streaming_unit_gen=streaming_unit_gen,

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            streamer_unit: Optional["BaseStreamer"] = None,
            streaming_unit_gen: bool = False,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            use_model_defaults: Optional[bool] = None,
            custom_generate: Optional[str] = None,
            **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head and CTC support.
            
            Additional parameters for CTC:
                streamer_unit (`BaseStreamer`, *optional*):
                    Streamer object for streaming generated units in CTC mode.
                streaming_unit_gen (`bool`, *optional*, defaults to `False`):
                    Whether to use streaming unit generation for CTC.
            """
            
            # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
            if custom_generate is not None:
                trust_remote_code = kwargs.pop("trust_remote_code", None)
                # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
                # they receive the same inputs as `generate`, only with `model` instead of `self`. They can access to
                # methods from `GenerationMixin` through `model`.
                global_keys_to_exclude = {"self", "kwargs"}
                generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
                generate_arguments.update(kwargs)

                custom_generate_function = self.load_custom_generate(
                    custom_generate, trust_remote_code=trust_remote_code, **kwargs
                )
                return custom_generate_function(model=self, **generate_arguments)

            # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
            tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
            assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

            generation_config, model_kwargs = self._prepare_generation_config(
                generation_config, use_model_defaults, **kwargs
            )
            self._validate_model_kwargs(model_kwargs.copy())
            self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

            # 2. Set generation parameters if not already defined
            if synced_gpus is None:
                synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

            accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
            requires_attention_mask = "encoder_outputs" not in model_kwargs
            kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

            # 3. Define model inputs
            inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
                inputs, generation_config.bos_token_id, model_kwargs
            )
            batch_size = inputs_tensor.shape[0]

            device = inputs_tensor.device
            self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

            # decoder-only models must use left-padding for batched generation.
            if not self.config.is_encoder_decoder:
                # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
                # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
                if (
                    generation_config._pad_token_tensor is not None
                    and batch_size > 1
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
                ):
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

            # 4. Define other model kwargs
            # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
            # generating the first new token or not, and we only want to use the embeddings for the first new token)
            if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
                generation_config.use_cache = True

            if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
                model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config, model_kwargs
                )
            elif kwargs_has_attention_mask:
                # TODO (joao): generalize this check with other types of inputs
                if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                    raise ValueError("`attention_mask` passed to `generate` must be 2D.")

            if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
                # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs, model_input_name, generation_config
                )

            # 5. Prepare `input_ids` which will be used for auto-regressive generation
            if self.config.is_encoder_decoder:
                input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size,
                    model_input_name=model_input_name,
                    model_kwargs=model_kwargs,
                    decoder_start_token_id=generation_config._decoder_start_token_tensor,
                    device=inputs_tensor.device,
                )
            else:
                input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

            if generation_config.token_healing:
                input_ids = self.heal_tokens(input_ids, tokenizer)

            if streamer is not None:
                streamer.put(input_ids.cpu())

            # 6. Prepare `max_length` depending on other stopping criteria.
            input_ids_length = input_ids.shape[1]
            has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
            has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
            generation_config = self._prepare_generated_length(
                generation_config=generation_config,
                has_default_max_length=has_default_max_length,
                has_default_min_length=has_default_min_length,
                model_input_name=model_input_name,
                inputs_tensor=inputs_tensor,
                input_ids_length=input_ids_length,
            )

            # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
            # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
            # dynamically overrides this value as it can need more than the last token logits
            if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
                model_kwargs["logits_to_keep"] = 1

            self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

            # 7. Prepare the cache.
            # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
            # - different models have a different cache name expected by the model (default = "past_key_values")
            # - `max_length`, prepared above, is used to determine the maximum cache length
            max_cache_length = generation_config.max_length - 1
            if (
                inputs_tensor.shape[1] != input_ids_length
                and model_input_name == "inputs_embeds"
                and not self.config.is_encoder_decoder
            ):
                max_cache_length += inputs_tensor.shape[1]
            self._prepare_cache_for_generation(
                generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
            )

            # 8. determine generation mode
            generation_mode = generation_config.get_generation_mode(assistant_model)

            if (streamer is not None or streamer_unit is not None) and (generation_config.num_beams > 1):
                raise ValueError(
                    "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
                )

            if self.device.type != input_ids.device.type:
                warnings.warn(
                    "You are calling .generate() with the `input_ids` being on a device type different"
                    f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                    f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                    " Please make sure that you have put `input_ids` to the"
                    f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                    " running `.generate()`.",
                    UserWarning,
                )

            # 9. prepare logits processors and stopping criteria
            prepared_logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
            prepared_stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
            )

            # Set model_kwargs `use_cache` so we can use it later in forward runs
            model_kwargs["use_cache"] = generation_config.use_cache

            # 10. go into different generation modes
            if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
                # 11. expand input_ids with `num_return_sequences` additional sequences per batch
                input_ids, model_kwargs = self._expand_inputs_for_generation(
                    input_ids=input_ids,
                    expand_size=generation_config.num_return_sequences,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    **model_kwargs,
                )

                # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
                # Use CTC-specific sampling if streaming_unit_gen is enabled
                if streaming_unit_gen:
                    result = self._sample_streaming_unit(
                        input_ids,
                        logits_processor=prepared_logits_processor,
                        stopping_criteria=prepared_stopping_criteria,
                        generation_config=generation_config,
                        synced_gpus=synced_gpus,
                        streamer=streamer,
                        streamer_unit=streamer_unit,
                        **model_kwargs,
                    )
                else:
                    result = self._sample(
                        input_ids,
                        logits_processor=prepared_logits_processor,
                        stopping_criteria=prepared_stopping_criteria,
                        generation_config=generation_config,
                        synced_gpus=synced_gpus,
                        streamer=streamer,
                        **model_kwargs,
                    )
            else:
                # For other generation modes, fall back to parent implementation
                # This includes beam search, assisted generation, etc.
                raise NotImplementedError(
                    f"Generation mode {generation_mode} is not implemented in GenerationWithCTC. "
                    "Only SAMPLE and GREEDY_SEARCH modes are supported with CTC features."
                )

            # Convert to legacy cache format if requested
            if (
                generation_config.return_legacy_cache is True
                and hasattr(result, "past_key_values")
                and getattr(result.past_key_values, "to_legacy_cache") is not None
            ):
                result.past_key_values = result.past_key_values.to_legacy_cache()
            
            return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        logits_warper = self._get_logits_warper(generation_config, device=input_ids.device) if generation_config.do_sample else None
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _sample_streaming_unit(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        streamer_unit: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        logits_warper = self._get_logits_warper(generation_config, device=input_ids.device) if generation_config.do_sample else None
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        generated_units = torch.tensor([])
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            # CTC-specific speech generation logic (commented out from original)
            # This is where you would implement your CTC unit generation
            # hidden_states = torch.cat([decoder_hidden_states[0][-1][:, -1:, :]] + [decoder_hidden_states[i][-1] for i in range(1, len(decoder_hidden_states))], dim=1)
            # ctc_pred = self.speech_generator.predict(hidden_states.squeeze(0))
            # cur_units = ctc_postprocess(ctc_pred, blank=self.model.config.unit_vocab_size)
            
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            
            # Stream units if available (commented out from original)
            # if streamer_unit is not None:
            #     for i in range(len(generated_units), len(cur_units)):
            #         streamer_unit.put(cur_units[i].unsqueeze(0))
            # generated_units = cur_units
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


def ctc_postprocess(tokens, blank):
    """
    Post-process CTC tokens by removing blanks and consecutive duplicates.
    
    Args:
        tokens: CTC output tokens
        blank: Blank token ID
        
    Returns:
        Processed tokens without blanks and duplicates
    """
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = torch.tensor([v for v in deduplicated_toks if v != blank])
    return hyp
