import customtkinter as ctk
import torch
import requests
import json
import os
from urllib.parse import quote
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from titan_mac import TitanMACBot

class WebSearch:
    def __init__(self):
        try:
            from googlesearch import search
            self.search_func = search
        except ImportError:
            print("Установите библиотеку: pip install googlesearch-python")
            self.search_func = None
            
    def search(self, query, num_results=4):
        """Поиск в Google"""
        if not self.search_func:
            return ["Установите библиотеку googlesearch-python командой:\npip install googlesearch-python"]
            
        try:
            # Выполняем поиск
            results = []
            for url in self.search_func(query, lang='ru', num_results=num_results):
                results.append(f"🔗 {url}")
            
            if results:
                return results
            return ["По вашему запросу ничего не найдено"]
            
        except Exception as e:
            return [f"Ошибка при поиске: {str(e)}"]

class ChatGUI:
    def __init__(self):
        # Настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Создание окна
        self.root = ctk.CTk()
        self.root.title("AI Chat")
        self.root.geometry("800x600")
        
        # Инициализация модели
        self.vocab_size = 1024
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        
        # Создаем бота и оптимизатор
        self.chatbot = TitanMACBot(vocab_size=self.vocab_size, d_model=self.d_model, 
                                 nhead=self.nhead, num_layers=self.num_layers)
        self.optimizer = optim.Adam(self.chatbot.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        
        # История диалогов и файлы для сохранения
        self.conversation_history = []
        self.memory_file = 'chat_memory.json'
        self.model_file = 'model.pth'
        
        # Загружаем историю и модель
        self.load_memory()
        self.load_model()
        
        # Состояние чата
        self.choosing_response = False
        self.current_user_input = ""
        self.current_responses = []
        self.response_buttons = []
        
        # Создаем поисковик
        self.web_search = WebSearch()
        
        # Создаем интерфейс
        self.setup_ui()
        
    def setup_ui(self):
        # Создаем основной контейнер
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Создаем область чата
        self.chat_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Создаем фрейм для ввода
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Поле ввода
        self.input_field = ctk.CTkEntry(
            input_frame,
            placeholder_text="Введите сообщение...",
            height=40,
            font=("Segoe UI", 12)
        )
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", self.send_message)
        
        # Кнопка отправки
        send_button = ctk.CTkButton(
            input_frame,
            text="Отправить",
            width=100,
            height=40,
            command=self.send_message
        )
        send_button.pack(side="right")
        
        # Фрейм для вариантов ответов
        self.responses_frame = ctk.CTkFrame(self.main_frame)
        self.responses_frame.pack(fill="x", padx=10)
        
    def append_to_chat(self, message, is_user=True):
        # Создаем фрейм для сообщения
        msg_frame = ctk.CTkFrame(self.chat_frame)
        msg_frame.pack(fill="x", padx=5, pady=5)
        
        # Добавляем метку с текстом сообщения
        prefix = "Вы: " if is_user else "Бот: "
        msg_label = ctk.CTkLabel(
            msg_frame,
            text=f"{prefix}{message}",
            font=("Segoe UI", 12),
            wraplength=600,
            justify="left"
        )
        msg_label.pack(anchor="w", padx=10, pady=5)
        
    def show_response_options(self, responses):
        # Очищаем предыдущие кнопки
        for button in self.response_buttons:
            button.destroy()
        self.response_buttons.clear()
        
        # Создаем новые кнопки с вариантами ответов
        for i, response in enumerate(responses):
            response_button = ctk.CTkButton(
                self.responses_frame,
                text=f"Вариант {i+1}:\n{response}",
                command=lambda r=response: self.select_response(r),
                height=60,
                font=("Segoe UI", 12),
                wraplength=500
            )
            response_button.pack(fill="x", padx=5, pady=5)
            self.response_buttons.append(response_button)
            
        # Добавляем кнопку поиска
        search_button = ctk.CTkButton(
            self.responses_frame,
            text="🔍 Поиск в интернете",
            command=self.search_web,
            height=40,
            font=("Segoe UI", 12)
        )
        search_button.pack(fill="x", padx=5, pady=5)
        self.response_buttons.append(search_button)
        
        # Добавляем кнопку своего варианта
        custom_button = ctk.CTkButton(
            self.responses_frame,
            text="✏️ Свой вариант ответа",
            command=self.show_custom_response_dialog,
            height=40,
            font=("Segoe UI", 12)
        )
        custom_button.pack(fill="x", padx=5, pady=5)
        self.response_buttons.append(custom_button)
        
    def show_custom_response_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Свой вариант ответа")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Создаем текстовое поле
        text_input = ctk.CTkTextbox(
            dialog,
            height=200,
            font=("Segoe UI", 12)
        )
        text_input.pack(fill="both", expand=True, padx=20, pady=(20, 10))
        
        def submit_response():
            response = text_input.get("1.0", "end-1c").strip()
            if response:
                self.select_response(response)
                dialog.destroy()
        
        # Кнопка подтверждения
        submit_button = ctk.CTkButton(
            dialog,
            text="Подтвердить",
            command=submit_response,
            height=40,
            font=("Segoe UI", 12)
        )
        submit_button.pack(pady=(0, 20))
        
        text_input.focus_set()
        
    def select_response(self, selected_response):
        # Добавляем выбранный ответ в чат
        self.append_to_chat(selected_response, is_user=False)
        
        # Очищаем кнопки
        for button in self.response_buttons:
            button.destroy()
        self.response_buttons.clear()
        
        # Обучаем бота на выбранном ответе
        self.learn_from_interaction(self.current_user_input, selected_response)
        
        # Сбрасываем состояние выбора
        self.choosing_response = False
        self.input_field.configure(state="normal")
        
    def send_message(self, event=None):
        if self.choosing_response:
            return
            
        message = self.input_field.get().strip()
        if not message:
            return
            
        self.input_field.delete(0, "end")
        self.append_to_chat(message, is_user=True)
        
        self.current_user_input = message
        self.current_responses = self.generate_response_options(message)
        
        self.show_response_options(self.current_responses)
        
        self.choosing_response = True
        self.input_field.configure(state="disabled")
        
    def search_web(self):
        search_results = self.web_search.search(self.current_user_input)
        self.show_response_options(search_results)
        
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
        """Пытается сгенерировать лучший ответ"""
        # Собираем все похожие диалоги из истории
        similar_dialogues = []
        for conv in self.conversation_history:
            if any(word in conv['user_input'].lower() for word in user_input.lower().split()):
                similar_dialogues.append(conv)
        
        # Генерируем варианты ответов с разными параметрами
        attempts = []
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
            
        # Если нет подходящих ответов, генерируем новый ответ с высокой температурой
        return self.chatbot.generate_response(user_input, temperature=1.2, max_length=150)
        
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
        
    def generate_response_options(self, user_input):
        """Генерирует варианты ответов на ввод пользователя"""
        # Пробуем разные параметры для генерации
        responses = []
        temperatures = [0.7, 1.0, 1.2]
        max_lengths = [50, 100, 150]
        
        for temp in temperatures:
            for length in max_lengths:
                response = self.chatbot.generate_response(user_input, temperature=temp, max_length=length)
                if response and len(response.strip()) >= 3:
                    responses.append(response)
        
        # Убираем дубликаты и пустые ответы
        responses = list(set(filter(None, responses)))
        responses = [r for r in responses if len(r.strip()) >= 3]
        
        if not responses:
            # Если не удалось сгенерировать ответы, пробуем с более высокой температурой
            response = self.chatbot.generate_response(user_input, temperature=1.5, max_length=200)
            if response and len(response.strip()) >= 3:
                responses = [response]
        
        return responses[:5]  # Возвращаем до 5 вариантов

def main():
    app = ChatGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main() 