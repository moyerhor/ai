import torch
import torch.nn as nn
import torch.nn.functional as F

class MACLayer(nn.Module):
    def __init__(self, hidden_size):
        super(MACLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        
        # Feed-forward
        ff_output = self.linear1(x)
        ff_output = F.relu(ff_output)
        ff_output = self.linear2(ff_output)
        
        # Residual connection и нормализация
        x = self.norm(x + ff_output)
        return x

class TitanMAC(nn.Module):
    def __init__(self, vocab_size=256, hidden_size=512, num_layers=4):
        super(TitanMAC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Эмбеддинг слов
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # MAC слои
        self.mac_layers = nn.ModuleList([
            MACLayer(hidden_size) for _ in range(num_layers)
        ])
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # Преобразуем входные токены в эмбеддинги
        x = self.embedding(x)
        
        # Проходим через MAC слои
        for layer in self.mac_layers:
            x = layer(x)
        
        # Выходной слой
        return self.output_layer(x)
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        self.eval()
        with torch.no_grad():
            # Преобразуем prompt в токены
            tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long)
            
            # Генерируем текст
            for _ in range(max_length):
                # Получаем предсказание следующего токена
                outputs = self(tokens)
                next_token_logits = outputs[0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Добавляем новый токен к последовательности
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                
                # Преобразуем токен в символ
                next_char = chr(next_token.item())
                if next_char == '\n':  # Останавливаемся при встрече символа новой строки
                    break
            
            # Преобразуем токены обратно в текст
            generated_text = ''.join([chr(t.item()) for t in tokens[0]])
            return generated_text

class ChatBot:
    def __init__(self, model_path, vocab_size=256, hidden_size=256):
        self.model = TitanMAC(vocab_size=vocab_size, hidden_size=hidden_size)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def generate_response(self, text, max_length=100):
        # Преобразуем входной текст в токены
        tokens = [min(ord(c) % self.model.vocab_size, self.model.vocab_size-1) for c in text]
        
        # Создаем тензор из токенов
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        # Генерируем ответ
        response_tokens = []
        with torch.no_grad():
            for _ in range(max_length):
                # Получаем предсказание следующего токена
                output = self.model(input_tensor)
                next_token = output[0, -1].argmax().item()
                
                # Добавляем токен к ответу
                response_tokens.append(next_token)
                
                # Если сгенерировали точку или перенос строки, заканчиваем
                if chr(next_token) in ['.', '\n']:
                    break
                
                # Обновляем входной тензор
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
        
        # Преобразуем токены обратно в текст
        response = ''.join([chr(t) for t in response_tokens])
        return response 