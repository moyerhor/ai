import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SimpleBot(nn.Module):
    def __init__(self, vocab_size=1024, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Простой энкодер
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        
        # Простой декодер
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, src):
        # Эмбеддинг входных токенов
        embedded = self.embedding(src)
        
        # Кодирование
        encoder_output, (hidden, cell) = self.encoder(embedded)
        
        # Декодирование
        decoder_output, _ = self.decoder(embedded, (hidden, cell))
        
        # Получаем логиты для каждого токена
        output = self.output(decoder_output)
        
        return output
    
    def generate(self, input_ids, max_length=100, temperature=0.7):
        self.eval()
        with torch.no_grad():
            # Начальный проход через энкодер
            embedded = self.embedding(input_ids)
            _, (hidden, cell) = self.encoder(embedded)
            
            # Используем последний токен входа как первый токен для декодера
            current_token = input_ids[:, -1:]
            generated = input_ids
            
            # Генерируем токены
            for _ in range(max_length):
                token_embedding = self.embedding(current_token)
                decoder_output, (hidden, cell) = self.decoder(token_embedding, (hidden, cell))
                
                # Получаем следующий токен
                logits = self.output(decoder_output)
                
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

class SimpleTokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        
        # Создаем простой словарь
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

class SimpleChatBot:
    def __init__(self, vocab_size=1024, hidden_size=256):
        self.model = SimpleBot(vocab_size=vocab_size, hidden_size=hidden_size)
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        
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