import logging
from .configuration_phys import PhysConfig

logger = logging.getLogger(__name__)

class GrayScottConfig(PhysConfig):
    """ This is the configuration class for the modeling of the gray-scott system.
        Args:
            vocab_size (:obj:`int`, optional, defaults to 50257):
                Vocabulary size of the GPT-2 model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.GPT2Model`.
            n_positions (:obj:`int`, optional, defaults to 1024):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            n_ctx (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the causal mask (usually same as n_positions).
            n_embd (:obj:`int`, optional, defaults to 768):
                Dimensionality of the embeddings and hidden states.
            n_layer (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            n_head (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            activation_function (:obj:`str`, optional, defaults to 'gelu'):
                Activation function selected in the list ["relu", "swish", "gelu", "tanh", "gelu_new"].
            resid_pdrop (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            embd_pdrop (:obj:`int`, optional, defaults to 0.1):
                The dropout ratio for the embeddings.
            attn_pdrop (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention.
            layer_norm_epsilon (:obj:`float`, optional, defaults to 1e-5):
                The epsilon to use in the layer normalization layers
            initializer_range (:obj:`float`, optional, defaults to 16):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            summary_type (:obj:`string`, optional, defaults to "cls_index"):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                Is one of the following options:
                - 'last' => take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                Add a projection after the vector extraction
            summary_activation (:obj:`string` or :obj:`None`, optional, defaults to :obj:`None`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                'tanh' => add a tanh activation to the output, Other => no activation.
            summary_proj_to_labels (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_first_dropout (:obj:`float`, optional, defaults to 0.1):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                Add a dropout before the projection and activation
    """

    model_type = "cylinder"

    def __init__(
        self,
        n_positions=128,
        n_ctx=128,
        n_embd=512,
        n_layer=2,
        n_head=32, # n_head must be a factor of n_embd
        state_dims=[2, 32, 32, 32],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.state_dims = state_dims
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
