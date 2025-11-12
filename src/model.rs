use crate::codec::DType;


pub enum ActivationType {
    GELU,
    SILU
}

pub enum LayerNormType{
    RMSNorm
}

pub struct Config{
    dim: usize, // transformer input & output dimension
    hidden_dim: usize, // dimension of hidden layer in feedforward network
    head_dim: usize,  // dimension of each attention head, usually dim / n_heads
    n_layers: usize,  // number of layers
    n_heads: usize, // number of attention query heads
    n_kv_heads: usize, // number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
    vocab_size: usize,  // vocabulary size
    max_seq_len: usize,  // max sequence length
    rope_theta: f32,  // RoPE theta
    rotary_dim: usize, // dimension of rotary position encoding (elements after that don't get rotated)
    norm_eps: f32, // epsilon for layer normalization
    act: ActivationType, // activation function
    norm_type: LayerNormType, // norm type
    qkv_clip: f32,  // clip qkv values to [-clip, clip]

    // Data type of the weights according to config, used
    // to safety check tensor dtype at initialization time.
    weight_dtype: DType,
}

pub struct Block<'a>{
    // weights for norms
    _rms_att_weight: &'a[f32], // (dim) rmsnorm weights
    _rms_ffn_weight: &'a[f32],  // (dim)

    // weights for self-attention matmuls
    _wq: &'a[u8], // (n_heads * head_dim, dim)
    _wk: &'a[u8], // (n_kv_heads * head_dim, dim)
    _wv: &'a[u8], // (n_kv_heads * head_dim, dim)
    _wo: &'a[u8], // (dim, n_heads * head_dim)
    
    // weights for ffn
    _w1: &'a[u8], // (n_experts?, hidden_dim, dim)
    _w2: &'a[u8], // (n_experts?, dim, hidden_dim)
    _w3: &'a[u8], // (n_experts?, hidden_dim, dim) - GLU weights

    // kv cache
    _key_cache : &'a[u8], // (seq_len, n_kv_heads * head_dim)
    _value_cache : &'a[u8], // (seq_len, n_kv_heads * head_dim)

}


pub struct InferenceState<'a>{
    _x: &'a[u8], // (dim,) - latest activation
    _xb:&'a[u8], // (dim,) - activation inside a residual branch
    _xb2: &'a[u8], // (dim,) - activation inside a residual branch (second slot)
    _hb: &'a[u8], // (hidden_dim,) - buffer for hidden dimension in feedforward network
    _hb2: &'a[u8],// (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
    _q: &'a[u8], // (n_heads * head_dim,) - query vectors for latest timestamp
    _k: &'a[u8], // (n_kv_heads * head_dim,) - key vectors for latest timestamp
    _v: &'a[u8], // (n_kv_heads * head_dim,) - value vectors for latest timestamp
    _att: &'a[u8], // (n_heads, seq_len) - buffer for attention scores
    
    // LM head
    _logits: &'a[u8], // (vocab_size,) - final output logits

}


struct Model<'a>{
    config: Config,
    blocks: Vec<Block<'a>>,
    // token embedding table
    token_embedding_table : &'a[u8], // (vocab_size, dim)
    // final norm
    rms_final_weight : &'a[f32], // (dim,)
    // classifier weights for the logits, on the last layer
    wcls: &'a[u8],// (vocab_size, dim)
    
    //TODO 
    //Model(YALMData& yalm);
}


//TODO create impl for structures