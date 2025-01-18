import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class TitanMAC(nn.Module):
    def __init__(self, vocab_size=1024, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Эмбеддинги и позиционное кодирование
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer энкодер
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Выходной слой
        self.output = nn.Linear(d_model, vocab_size)
        
        # Инициализация весов
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_mask=None):
        # Эмбеддинг и позиционное кодирование
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        
        # Проход через трансформер
        output = self.transformer_encoder(src, src_mask)
        
        # Получаем логиты для каждого токена
        output = self.output(output)
        
        return output
    
    def generate(self, input_ids, max_length=100, temperature=0.7):
        self.eval()
        with torch.no_grad():
            # Начальный проход через модель
            src = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            src = self.pos_encoder(src)
            memory = self.transformer_encoder(src)
            
            # Используем последний токен входа как первый токен для генерации
            current_token = input_ids[:, -1:]
            generated = input_ids
            
            # Генерируем токены
            for _ in range(max_length):
                # Эмбеддинг текущего токена
                token_embedding = self.embedding(current_token) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
                token_embedding = self.pos_encoder(token_embedding)
                
                # Получаем следующий токен
                decoder_output = self.transformer_encoder(token_embedding)
                logits = self.output(decoder_output[:, -1:])
                
                # Применяем температуру и softmax
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Фильтруем нежелательные токены
                probs[0, 0, :32] = 0  # Убираем контрольные символы
                probs = probs / probs.sum()  # Нормализуем
                
                # Выбираем следующий токен
                next_token = torch.multinomial(probs[0, 0], 1).unsqueeze(0)
                
                # Добавляем к результату
                generated = torch.cat([generated, next_token], dim=1)
                current_token = next_token
                
                # Проверяем на токен конца последовательности
                if next_token.item() == self.vocab_size - 1 or next_token.item() == 3:  # EOS token
                    break
                    
        return generated

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Создаем матрицу позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Tokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        
        # Создаем словарь
        self.char_to_idx = {chr(i): i for i in range(32, 127)}
        self.idx_to_char = {i: chr(i) for i in range(32, 127)}
        
        # Добавляем специальные токены
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Обновляем словари
        self.char_to_idx.update(self.special_tokens)
        self.idx_to_char.update({v: k for k, v in self.special_tokens.items()})
        
    def encode(self, text: str) -> List[int]:
        tokens = [self.special_tokens['<BOS>']]
        for char in text:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.special_tokens['<UNK>'])
        tokens.append(self.special_tokens['<EOS>'])
        return tokens
        
    def decode(self, tokens: List[int]) -> str:
        text = []
        for token in tokens:
            if token in [self.special_tokens['<PAD>'], self.special_tokens['<BOS>'], self.special_tokens['<UNK>']]:
                continue
            if token == self.special_tokens['<EOS>']:
                break
            if token in self.idx_to_char:
                text.append(self.idx_to_char[token])
        return ''.join(text)

class TitanMACBot:
    def __init__(self, vocab_size=1024, d_model=256, nhead=8, num_layers=4):
        self.model = TitanMAC(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.tokenizer = Tokenizer(vocab_size=vocab_size)
        
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
    def generate_response(self, text: str, max_length: int = 100, temperature: float = 0.7) -> str:
        self.model.eval()
        with torch.no_grad():
            # Кодируем входной текст
            tokens = self.tokenizer.encode(text)
            input_ids = torch.tensor([tokens]).long()
            
            # Пробуем сгенерировать ответ несколько раз с разной температурой
            attempts = [
                (0.7, max_length),  # Стандартная попытка
                (1.0, max_length),  # Более креативная
                (0.5, max_length * 2),  # Более длинная и консервативная
                (1.2, max_length // 2),  # Короткая но более случайная
            ]
            
            for temp, length in attempts:
                # Генерируем ответ с текущими параметрами
                output_ids = self.model.generate(input_ids, max_length=length, temperature=temp)
                response = self.tokenizer.decode(output_ids[0].tolist())
                
                # Если ответ не пустой и содержит хотя бы 2 символа, возвращаем его
                if response and len(response.strip()) >= 2 and not response.isspace():
                    return response
            
            # Если все попытки не удались, генерируем простой ответ
            simple_responses = ["Привет!", "Да", "Нет", "Хорошо", "Интересно", "Понятно"]
            return simple_responses[torch.randint(0, len(simple_responses), (1,)).item()] 