from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from zero_shot import CTClipInference
from src.args import parse_arguments
from head_utils import assert_head_checkpoint_compatible, require_head_aware_cli_args
from text_model_utils import build_text_encoder, build_tokenizer

args = parse_arguments()
require_head_aware_cli_args(args, ("pretrained", "data_folder", "reports_file", "labels", "save"))
assert_head_checkpoint_compatible(args)
if args.target_depth <= 0 or args.target_depth % 10 != 0:
    raise ValueError("--target-depth must be a positive multiple of 10 for CTViT temporal patches.")

target_shape = (480, 480, args.target_depth)
target_spacing = (1.5 * 240 / args.target_depth, 0.75, 0.75)
channels = 4 if args.head else 1

tokenizer = build_tokenizer(head=args.head)
text_encoder = build_text_encoder(head=args.head, tokenizer=tokenizer)

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8,
    channels = channels
)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = text_encoder.config.hidden_size,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False,
    tokenizer = tokenizer

)

clip.load(args.pretrained)

inference = CTClipInference(
    clip,
    data_folder = args.data_folder,
    reports_file= args.reports_file,
    meta_file = args.meta_file,
    labels = args.labels,
    results_folder = args.save,
    head = args.head,
    target_spacing = target_spacing,
    target_shape = target_shape,
    preprocessed_nifti = args.preprocessed_nifti,
)

inference.infer()
