import torch
import torch.nn as nn
from models.model_emb import ModelEmb
from models.segmentor_net import HardNetSegmentor
from packaging import version

def get_model(args, backbone_weight_dir='', embedding_dim=256, use_tanh=True):
    backbone_type = args['backbone_type']
    if backbone_type != 'HardNet': raise ValueError(f"Unsupported backbone_type='{backbone_type}'. Only 'HardNet' is currently supported.")
    try: model = ModelEmb(args=args, backbone_weight_dir=backbone_weight_dir, embedding_dim=embedding_dim, use_tanh=use_tanh)
    except Exception as e: raise RuntimeError(f"Failed to build ModelEmb (backbone_type={backbone_type}): {e}") from e

    ckpt_path = args['ckpt_path']
    if ckpt_path is not None:
        try: ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            try: ckpt = torch.load(ckpt_path, map_location='cpu')
            except Exception as e: raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {e}") from e
        except Exception as e: raise RuntimeError(f"Failed to load checkpoint '{ckpt_path}': {e}") from e

        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt: state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
            elif 'model' in ckpt and isinstance(ckpt['model'], nn.Module): state_dict = ckpt['model'].state_dict()
            else: state_dict = ckpt
        else: state_dict = ckpt.state_dict()

        strict = not args.get('load_ckpt_single_dict', False)

        try: result = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e: raise RuntimeError(f"Failed to load state_dict into model: {e}") from e

        print(f"[get_model] Checkpoint loaded from '{ckpt_path}' (strict={strict})")

        if not strict:
            if result.missing_keys:
                print(f"[get_model]   Missing keys   : {len(result.missing_keys)}")
                for key in result.missing_keys[:5]: print(f"   - {key}")
            if result.unexpected_keys:
                print(f"[get_model]   Unexpected keys: {len(result.unexpected_keys)}")
                for key in result.unexpected_keys[:5]: print(f"   - {key}")
    print()        
    return model

def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_empty, _ = sam.prompt_encoder(points=None, boxes=None, masks=None)

    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_empty,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False)
    return low_res_masks

def get_standard_model(args, backbone_weight_dir='', out_ch=256, use_tanh=True):
    if args['net']=='HardNetSegmentor': model = HardNetSegmentor(args, segmentor=args['net_segmentor'], backbone_weight_dir=backbone_weight_dir, out_ch=out_ch, use_tanh=use_tanh)
    else: raise ValueError(f"Model not implemented for net='{args['net']}' Supported: net='HardNetSegmentor'")
    
    ckpt_dir = args['ckpt_path']
    if ckpt_dir is not None:
        print(f"[get_standard_model] Loading checkpoint: {ckpt_dir}")

        try:
            torch_version = version.parse(torch.__version__)
            if torch_version >= version.parse('1.13.0'): ckpt_model = torch.load(ckpt_dir, map_location='cpu', weights_only=False)
            else: ckpt_model = torch.load(ckpt_dir, map_location='cpu')
        except Exception as e: raise RuntimeError(f"Failed to load checkpoint '{ckpt_dir}': {e}") from e
 
        if not args['use_sam'] and args['segmentor_finetune_backbone']: strict = False
        else: strict = True

        if isinstance(ckpt_model, nn.Module): state_dict = ckpt_model.state_dict()
        elif isinstance(ckpt_model, dict):
            state_dict = None

            for key in ("model_state_dict", "state_dict"):
                if key in ckpt_model:
                    state_dict = ckpt_model[key]
                    break

            if state_dict is None and isinstance(ckpt_model.get("model"), nn.Module): state_dict = ckpt_model["model"].state_dict()

            if state_dict is None: state_dict = ckpt_model
        else: raise TypeError(f"Unsupported checkpoint format: expected a model, state dictionary, or checkpoint dictionary; got {type(ckpt_model).__name__}.")

        if isinstance(state_dict, nn.Module): state_dict = state_dict.state_dict()
        if not isinstance(state_dict, dict): raise TypeError("The extracted model state must be a dictionary.")

        try: result = model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e: raise RuntimeError(f"Failed to load state_dict into model: {e}") from e
        print(f"[get_standard_model] Checkpoint loaded (strict={strict})")

        if not strict:
            if result.missing_keys:
                print(f"[get_standard_model]   Missing keys   : {len(result.missing_keys)}")
                for key in result.missing_keys[:5]: print(f"   - {key}")
                if len(result.missing_keys) > 5: print(f"   ... ({len(result.missing_keys) - 5} more)")
                if all(k.startswith('segmentor.') for k in result.missing_keys): print(f"[get_standard_model]   Note: all missing keys belong to 'segmentor.*' (new aggregator in Phase 2, initialized from scratch)")
            else: print(f"[get_standard_model]   Missing keys   : 0 ✅")

            if result.unexpected_keys:
                print(f"[get_standard_model]   Unexpected keys: {len(result.unexpected_keys)}")
                for key in result.unexpected_keys[:5]: print(f"   - {key}")
            else: print(f"[get_standard_model]   Unexpected keys: 0 ✅")    

    print()
    return model