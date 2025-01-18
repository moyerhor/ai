import torch
import torch.nn as nn
import torch.optim as optim
from titan_llm import SimpleChatBot
import os
import json
from datetime import datetime

class InteractiveLearningBot:
    def __init__(self, vocab_size=1024, hidden_size=256):
        self.chatbot = SimpleChatBot(vocab_size=vocab_size, hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.chatbot.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # игнорируем PAD токен
        self.conversation_history = []
        self.memory_file = 'chat_memory.json'
        self.model_file = 'model.pth'
        
        # Загружаем историю общения и модель если они есть
        self.load_memory()
        self.load_model()
        
    def save_memory(self):
        """Сохраняем историю общения"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            
    def load_memory(self):
        """Загружаем историю общения"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                print(f"Загружено {len(self.conversation_history)} диалогов из памяти")
            except:
                print("Не удалось загрузить память, начинаем с чистого листа")
                self.conversation_history = []
                
    def save_model(self):
        """Сохраняем модель"""
        self.chatbot.save(self.model_file)
        
    def load_model(self):
        """Загружаем модель"""
        if os.path.exists(self.model_file):
            try:
                self.chatbot.load(self.model_file)
                print("Модель успешно загружена")
            except:
                print("Не удалось загрузить модель, начинаем с начала")
    
    def try_generate_better_response(self, user_input, wrong_response):
        """Пытается сгенерировать лучший ответ на основе контекста"""
        # Список возможных ответов для разных типов вопросов
        responses = {
            'кто': [
                'Я чат-бот, который учится общаться',
                'Я ваш виртуальный собеседник',
                'Я искусственный интеллект в процессе обучения',
                'Я простой чат-бот, который хочет научиться общаться'
            ],
            'как': [
                'Хорошо, спасибо',
                'Нормально',
                'Отлично',
                'В процессе обучения'
            ],
            'привет': [
                'Здравствуйте',
                'Привет',
                'Добрый день',
                'Рад вас видеть'
            ],
            'default': [
                'Интересный вопрос',
                'Давайте поговорим об этом',
                'Мне нужно подумать над этим',
                'Это сложный вопрос для меня'
            ]
        }

        # Определяем тип вопроса
        question_type = 'default'
        user_input_lower = user_input.lower()
        for key in responses.keys():
            if key in user_input_lower:
                question_type = key
                break

        # Пробуем разные варианты ответов
        attempts = []
        
        # Добавляем специфичные ответы для данного типа вопроса
        attempts.extend(responses[question_type])
        
        # Пробуем сгенерировать новые ответы с разными параметрами
        temperatures = [0.5, 0.7, 1.0, 1.2]
        max_lengths = [50, 100, 150]
        
        for temp in temperatures:
            for length in max_lengths:
                response = self.chatbot.generate_response(user_input, temperature=temp, max_length=length)
                if response and response != wrong_response and len(response.strip()) >= 3:
                    attempts.append(response)

        # Убираем дубликаты и пустые ответы
        attempts = list(set(filter(None, attempts)))
        attempts = [r for r in attempts if r != wrong_response and len(r.strip()) >= 3]

        if attempts:
            return attempts[torch.randint(0, len(attempts), (1,)).item()]
        return "Извините, я пока учусь отвечать правильно"

    def learn_from_interaction(self, user_input, wrong_response):
        """Обучение на одном взаимодействии"""
        # Генерируем новый ответ
        better_response = self.try_generate_better_response(user_input, wrong_response)
        
        # Кодируем входной текст и новый ответ
        input_tokens = self.chatbot.tokenizer.encode(user_input)
        target_tokens = self.chatbot.tokenizer.encode(better_response)
        
        # Создаем тензоры
        input_tensor = torch.tensor([input_tokens]).long()
        target_tensor = torch.tensor([target_tokens]).long()
        
        # Увеличиваем learning rate для быстрого обучения
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 10
            
        # Обучаем модель
        self.optimizer.zero_grad()
        output = self.chatbot.model(input_tensor)
        
        # Обрезаем выход до длины цели
        min_len = min(output.size(1), target_tensor.size(1))
        output = output[:, :min_len, :]
        target_tensor = target_tensor[:, :min_len]
        
        # Считаем loss
        loss = self.loss_fn(output.view(-1, self.chatbot.model.vocab_size), target_tensor.view(-1))
        loss.backward()
        self.optimizer.step()
        
        # Возвращаем learning rate обратно
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1
        
        # Сохраняем взаимодействие в истории
        self.conversation_history.append({
            'user_input': user_input,
            'bot_response': better_response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'loss': float(loss.item()),
            'was_corrected': True
        })
        
        # Сохраняем обновленную историю и модель
        self.save_memory()
        self.save_model()
        
        return better_response, float(loss.item())
    
    def generate_response(self, text):
        """Генерация ответа"""
        # Ищем похожие вопросы в истории
        similar_responses = []
        for conv in self.conversation_history:
            if text.lower() in conv['user_input'].lower() or conv['user_input'].lower() in text.lower():
                similar_responses.append(conv['bot_response'])
        
        # Если нашли похожие ответы, используем их как основу
        if similar_responses and torch.rand(1).item() < 0.7:
            base_response = similar_responses[torch.randint(0, len(similar_responses), (1,)).item()]
            # Иногда добавляем вариативность к сохраненному ответу
            if torch.rand(1).item() < 0.3 and len(base_response) > 0:
                if not base_response.endswith(('!', '?', '.')):
                    base_response += ['!', '.'][torch.randint(0, 2, (1,)).item()]
                if torch.rand(1).item() < 0.5:
                    base_response = base_response.capitalize()
            return base_response
        
        # Генерируем новый ответ с разными параметрами
        response = self.chatbot.generate_response(text, temperature=0.7)
        
        # Если ответ слишком короткий, пробуем с другой температурой
        if len(response.strip()) < 3:
            response = self.chatbot.generate_response(text, temperature=1.2, max_length=150)
            
        return response

def main():
    print("\nПростой интерактивный чат-бот с самообучением")
    print("Команды:")
    print("- 'выход' - завершить общение")
    print("- 'память' - показать историю общения")
    print("- 'статистика' - показать статистику обучения")
    print("- 'сброс' - сбросить модель и начать обучение заново\n")
    
    bot = InteractiveLearningBot()
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if user_input.lower() == 'выход':
            print("До свидания! Сохраняю память и модель...")
            bot.save_memory()
            bot.save_model()
            break
            
        elif user_input.lower() == 'память':
            print("\nИстория общения:")
            for i, conv in enumerate(bot.conversation_history[-10:], 1):
                print(f"\n{i}. {conv['timestamp']}")
                print(f"Вы: {conv['user_input']}")
                print(f"Бот: {conv['bot_response']}")
                print(f"Ошибка: {conv['loss']:.4f}")
            continue
            
        elif user_input.lower() == 'статистика':
            if bot.conversation_history:
                losses = [conv['loss'] for conv in bot.conversation_history]
                print(f"\nВсего взаимодействий: {len(losses)}")
                print(f"Средняя ошибка: {sum(losses)/len(losses):.4f}")
                print(f"Минимальная ошибка: {min(losses):.4f}")
                print(f"Максимальная ошибка: {max(losses):.4f}")
            else:
                print("\nПока нет данных для статистики")
            continue
            
        elif user_input.lower() == 'сброс':
            confirm = input("Вы уверены? Все обучение будет потеряно (да/нет): ").strip().lower()
            if confirm == 'да':
                bot = InteractiveLearningBot()
                if os.path.exists(bot.model_file):
                    os.remove(bot.model_file)
                if os.path.exists(bot.memory_file):
                    os.remove(bot.memory_file)
                print("Модель сброшена")
            continue
            
        # Генерируем ответ
        response = bot.generate_response(user_input)
        print(f"Бот: {response}")
        
        # Спрашиваем, правильный ли ответ
        correction = input("Ответ правильный? (да/нет): ").strip().lower()
        
        if correction == 'нет':
            print("Пробую придумать лучший ответ...")
            better_response, loss = bot.learn_from_interaction(user_input, response)
            print(f"Бот: {better_response}")
            print(f"Ошибка обучения: {loss:.4f}")

if __name__ == "__main__":
    main() 