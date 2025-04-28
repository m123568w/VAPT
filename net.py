import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from dall_e import map_pixels, unmap_pixels, load_model

print(torch.__version__)
print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KnowledgeTransformer(vit, x, contexts):
    x = vit.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + vit.positional_embedding.to(x.dtype)
    x = vit.ln_pre(x)
    
    contexts = contexts.repeat(x.size()[0], 1, 1).to(device)
    x = torch.cat((x, contexts), 1).half()
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = vit.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = vit.ln_post(x[:, 0, :])

    if vit.proj is not None:
        x = x @ vit.proj
    return x


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size_q, input_size_kv, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size_q, self.all_head_size)
        self.key = nn.Linear(input_size_kv, self.all_head_size)
        self.value = nn.Linear(input_size_kv, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print(new_x_shape)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_q, input_k, input_v):
        query_layer = self.transpose_for_scores(self.query(input_q))
        key_layer = self.transpose_for_scores(self.key(input_k))
        value_layer = self.transpose_for_scores(self.value(input_v))
        
        # Cross-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context = context_layer.view(*new_context_layer_shape)
        # hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return context

    
class LinearClassifier(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super(LinearClassifier, self).__init__() 
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x): 
        x = self.fc(x)
        # print(x.size())
        return F.log_softmax(x, dim=1)
        
        
class TextEncoder(nn.Module):
    def __init__(self, model):
        '''
        model: CLIP model
        '''
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.dtype = model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2).to(torch.float16)  # NLD -> LND
        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
        
class PromptLearner(nn.Module):
    def __init__(self, cls_labels, model):
        '''
        cls_labels: list of class labels
        model: CLIP model
        '''
        super().__init__()
        n_context = 16  # 与coop一致
        context_dim = model.ln_final.weight.shape[0]  # 512
        len_knowledge = len(cls_labels)
        
        context_vectors = torch.empty(1, n_context, context_dim)
        nn.init.normal_(context_vectors, std=0.02)
        context_vectors = context_vectors.repeat(len_knowledge, 1, 1).to(device)
        
        self.context = nn.Parameter(context_vectors)
        
        prompt_prefix = " ".join(["X"] * n_context)
        prompts = [prompt_prefix + " " + kl for kl in cls_labels]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = model.token_embedding(tokenized_prompts)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # Start Of the Sentence
        self.register_buffer("token_suffix", embedding[:, 1 + n_context :, :])  # Expert Knowledge, End of Sentence
        
        self.n_context = n_context
        self.tokenized_prompts = tokenized_prompts
        
    def forward(self):
        context = self.context
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, context, suffix], dim=1)
        
        return prompts, context
        

class MyKnowledgeNet(nn.Module):
    def __init__(self, cls_labels, model, num_classes):
        super(MyKnowledgeNet, self).__init__()
        
        enc = load_model("./model/encoder.pkl", device)
        dec = load_model("./model/decoder.pkl", device)
        params = enc.state_dict()  # 提取出的visual codebook的参数
        self.vc_weight = params["blocks.output.conv.w"]
        self.vc_weight = self.vc_weight.squeeze(2).squeeze(2).unsqueeze(0)
        
        num_attention_heads = 8
        input_size_q = 512
        input_size_kv = 2048
        hidden_size = 768
        hidden_dropout_prob = 0.1
        self.self_attention = SelfAttention(num_attention_heads, input_size_q, input_size_kv, hidden_size, hidden_dropout_prob).to(device)
        
        input_dim = 512
        output_dim = num_classes
        self.linear = LinearClassifier(input_dim, output_dim).to(device)
        
        self.prompt_learner = PromptLearner(cls_labels).to(device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_encoder = TextEncoder()
        
        self.vt = model.visual.to(device)

    def forward(self, images):
        prompts, context = self.prompt_learner()
        
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features.unsqueeze(0).to(torch.float32)
        context_vector = context.mean(dim=0)
        context_vector = context_vector.unsqueeze(0).to(torch.float32)
        k = self.self_attention(context_vector, self.vc_weight, self.vc_weight)
        i = KnowledgeTransformer(self.vt, images, k).float()
        i = self.linear(i)
        return i