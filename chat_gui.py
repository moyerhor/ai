import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext
from interactive_chat import InteractiveLearningBot
import torch

class DarkTheme:
    # Основные цвета
    BG_COLOR = "#1e1e1e"  # Более темный фон
    FG_COLOR = "#e0e0e0"  # Более мягкий белый для текста
    
    # Цвета для кнопок
    BUTTON_BG = "#2d2d2d"  # Темный фон кнопок
    BUTTON_BG_HOVER = "#3d3d3d"  # Цвет при наведении
    BUTTON_FG = "#e0e0e0"  # Цвет текста кнопок
    
    # Цвета для полей ввода
    ENTRY_BG = "#2d2d2d"  # Фон полей ввода
    ENTRY_FG = "#e0e0e0"  # Цвет текста в полях ввода
    
    # Цвета для чата
    CHAT_BG = "#1e1e1e"  # Фон области чата
    CHAT_FG = "#e0e0e0"  # Цвет текста чата
    CHAT_USER_MSG = "#4a9eff"  # Цвет сообщений пользователя
    CHAT_BOT_MSG = "#45c937"  # Цвет сообщений бота

class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Чат с ИИ")
        self.root.geometry("800x600")
        
        # Применяем темную тему
        self.apply_dark_theme()
        
        # Инициализируем бота
        self.bot = InteractiveLearningBot()
        
        # Создаем и размещаем компоненты
        self.create_widgets()
        
        # Состояние выбора ответа
        self.choosing_response = False
        self.response_buttons = []
        self.current_responses = []
        
    def apply_dark_theme(self):
        self.root.configure(bg=DarkTheme.BG_COLOR)
        
        style = ttk.Style()
        
        # Настройка стиля фрейма
        style.configure(
            "Dark.TFrame",
            background=DarkTheme.BG_COLOR,
        )
        
        # Настройка стиля кнопок
        style.configure(
            "Dark.TButton",
            background=DarkTheme.BUTTON_BG,
            foreground=DarkTheme.BUTTON_FG,
            borderwidth=0,
            focuscolor=DarkTheme.BUTTON_BG_HOVER,
            lightcolor=DarkTheme.BUTTON_BG,
            darkcolor=DarkTheme.BUTTON_BG,
            relief="flat",
            padding=10
        )
        
        # Настройка стиля полей ввода
        style.configure(
            "Dark.TEntry",
            fieldbackground=DarkTheme.ENTRY_BG,
            foreground=DarkTheme.ENTRY_FG,
            borderwidth=0,
            relief="flat",
            padding=5
        )
        
        # Настройка стиля при наведении на кнопку
        style.map(
            "Dark.TButton",
            background=[("active", DarkTheme.BUTTON_BG_HOVER)],
            foreground=[("active", DarkTheme.BUTTON_FG)]
        )
        
    def create_widgets(self):
        # Создаем основной контейнер
        main_container = ttk.Frame(self.root, padding="20", style="Dark.TFrame")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Создаем фрейм для чата и скроллбара
        chat_frame = ttk.Frame(main_container, style="Dark.TFrame")
        chat_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="nsew")
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Создаем область чата с отступами и закругленными углами
        self.chat_area = tk.Text(
            chat_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=20,
            bg=DarkTheme.CHAT_BG,
            fg=DarkTheme.CHAT_FG,
            insertbackground=DarkTheme.FG_COLOR,
            font=("Segoe UI", 10),
            relief="flat",
            padx=10,
            pady=10,
            selectbackground=DarkTheme.BUTTON_BG_HOVER,
            selectforeground=DarkTheme.FG_COLOR,
        )
        self.chat_area.grid(row=0, column=0, sticky="nsew")
        
        # Создаем скроллбар для чата
        chat_scrollbar = tk.Scrollbar(
            chat_frame, 
            orient="vertical", 
            command=self.chat_area.yview,
            bg=DarkTheme.BG_COLOR,
            troughcolor=DarkTheme.ENTRY_BG,
            activebackground=DarkTheme.BUTTON_BG_HOVER,
        )
        chat_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_area.configure(yscrollcommand=chat_scrollbar.set)
        
        # Настраиваем тег для сообщений пользователя
        self.chat_area.tag_configure("user_msg", foreground=DarkTheme.CHAT_USER_MSG)
        # Настраиваем тег для сообщений бота
        self.chat_area.tag_configure("bot_msg", foreground=DarkTheme.CHAT_BOT_MSG)
        self.chat_area.config(state='disabled')
        
        # Создаем поле ввода с закругленными углами
        self.input_field = tk.Entry(
            main_container, 
            width=60,
            bg=DarkTheme.ENTRY_BG,
            fg=DarkTheme.ENTRY_FG,
            insertbackground=DarkTheme.FG_COLOR,
            font=("Segoe UI", 10),
            relief="flat",
            bd=10,
            selectbackground=DarkTheme.BUTTON_BG_HOVER,
            selectforeground=DarkTheme.FG_COLOR,
        )
        self.input_field.grid(row=1, column=0, pady=(0, 20), sticky="ew")
        self.input_field.bind("<Return>", self.send_message)
        
        # Создаем кнопку отправки
        send_button = ttk.Button(
            main_container, 
            text="Отправить", 
            command=self.send_message,
            style="Dark.TButton"
        )
        send_button.grid(row=1, column=1, pady=(0, 20), padx=(10, 0))
        
        # Создаем фрейм для карточек ответов
        self.responses_frame = ttk.Frame(main_container, style="Dark.TFrame")
        self.responses_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        # Настраиваем растяжение
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=3)
        main_container.columnconfigure(1, weight=1)
        
    def append_to_chat(self, message, is_user=True):
        self.chat_area.config(state='normal')
        prefix = "Вы: " if is_user else "Бот: "
        tag = "user_msg" if is_user else "bot_msg"
        
        self.chat_area.insert(tk.END, prefix, tag)
        self.chat_area.insert(tk.END, message + "\n\n")  # Добавляем дополнительный перенос строки
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')
        
    def generate_response_options(self, user_input):
        # Генерируем 4 разных варианта ответа
        responses = []
        temperatures = [0.5, 0.7, 1.0, 1.2]
        
        # Получаем базовый ответ
        base_response = self.bot.generate_response(user_input)
        responses.append(base_response)
        
        # Генерируем дополнительные варианты
        while len(responses) < 4:
            # Пробуем сгенерировать новый ответ
            temp = temperatures[len(responses) - 1]
            response = self.bot.try_generate_better_response(user_input, base_response)
            
            # Добавляем только если это новый уникальный ответ
            if response not in responses:
                responses.append(response)
        
        return responses
        
    def show_custom_response_dialog(self):
        # Создаем диалоговое окно для ввода своего ответа
        dialog = tk.Toplevel(self.root)
        dialog.title("Свой вариант ответа")
        dialog.geometry("400x200")
        dialog.configure(bg=DarkTheme.BG_COLOR)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Создаем фрейм для текстового поля и скроллбара
        text_frame = ttk.Frame(dialog, style="Dark.TFrame")
        text_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Создаем текстовое поле для ввода
        text_input = tk.Text(
            text_frame,
            wrap=tk.WORD,
            width=40,
            height=5,
            bg=DarkTheme.ENTRY_BG,
            fg=DarkTheme.ENTRY_FG,
            insertbackground=DarkTheme.FG_COLOR,
            font=("Segoe UI", 10),
            relief="flat",
            padx=10,
            pady=10,
            selectbackground=DarkTheme.BUTTON_BG_HOVER,
            selectforeground=DarkTheme.FG_COLOR,
        )
        text_input.grid(row=0, column=0, sticky="nsew")
        
        # Создаем скроллбар для текстового поля
        text_scrollbar = tk.Scrollbar(
            text_frame, 
            orient="vertical", 
            command=text_input.yview,
            bg=DarkTheme.BG_COLOR,
            troughcolor=DarkTheme.ENTRY_BG,
            activebackground=DarkTheme.BUTTON_BG_HOVER,
        )
        text_scrollbar.grid(row=0, column=1, sticky="ns")
        text_input.configure(yscrollcommand=text_scrollbar.set)
        
        def submit_response():
            response = text_input.get("1.0", tk.END).strip()
            if response:
                self.select_response(response)
                dialog.destroy()
        
        # Кнопка подтверждения
        submit_button = ttk.Button(
            dialog,
            text="Подтвердить",
            command=submit_response,
            style="Dark.TButton"
        )
        submit_button.pack(pady=(0, 20))
        
        # Фокус на поле ввода
        text_input.focus_set()
        
    def show_response_options(self, responses):
        # Очищаем предыдущие кнопки
        for btn in self.response_buttons:
            btn.destroy()
        self.response_buttons.clear()
        
        # Создаем новые кнопки с вариантами ответов
        for i, response in enumerate(responses):
            # Создаем фрейм для кнопки с темным фоном
            btn_frame = tk.Frame(
                self.responses_frame,
                bg=DarkTheme.BUTTON_BG,
                padx=10,
                pady=10
            )
            btn_frame.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="ew")
            
            # Создаем метку с текстом внутри фрейма
            label = tk.Label(
                btn_frame,
                text=f"Вариант {i+1}:\n{response}",
                bg=DarkTheme.BUTTON_BG,
                fg=DarkTheme.BUTTON_FG,
                font=("Segoe UI", 10),
                wraplength=300,  # Ограничиваем ширину текста
                justify=tk.LEFT,
                padx=5,
                pady=5
            )
            label.pack(fill=tk.BOTH, expand=True)
            
            # Добавляем обработчики событий для эффекта при наведении
            def on_enter(e, frame=btn_frame, lbl=label):
                frame.configure(bg=DarkTheme.BUTTON_BG_HOVER)
                lbl.configure(bg=DarkTheme.BUTTON_BG_HOVER)
                
            def on_leave(e, frame=btn_frame, lbl=label):
                frame.configure(bg=DarkTheme.BUTTON_BG)
                lbl.configure(bg=DarkTheme.BUTTON_BG)
                
            def on_click(r=response):
                self.select_response(r)
            
            btn_frame.bind("<Enter>", on_enter)
            btn_frame.bind("<Leave>", on_leave)
            btn_frame.bind("<Button-1>", lambda e, r=response: on_click(r))
            label.bind("<Enter>", on_enter)
            label.bind("<Leave>", on_leave)
            label.bind("<Button-1>", lambda e, r=response: on_click(r))
            
            self.response_buttons.append((btn_frame, label))
            
        # Добавляем кнопку для своего варианта
        custom_frame = tk.Frame(
            self.responses_frame,
            bg=DarkTheme.BUTTON_BG,
            padx=10,
            pady=10
        )
        custom_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        custom_label = tk.Label(
            custom_frame,
            text="Свой вариант ответа",
            bg=DarkTheme.BUTTON_BG,
            fg=DarkTheme.BUTTON_FG,
            font=("Segoe UI", 10),
            padx=5,
            pady=5
        )
        custom_label.pack(fill=tk.BOTH, expand=True)
        
        def on_custom_enter(e):
            custom_frame.configure(bg=DarkTheme.BUTTON_BG_HOVER)
            custom_label.configure(bg=DarkTheme.BUTTON_BG_HOVER)
            
        def on_custom_leave(e):
            custom_frame.configure(bg=DarkTheme.BUTTON_BG)
            custom_label.configure(bg=DarkTheme.BUTTON_BG)
            
        custom_frame.bind("<Enter>", on_custom_enter)
        custom_frame.bind("<Leave>", on_custom_leave)
        custom_frame.bind("<Button-1>", lambda e: self.show_custom_response_dialog())
        custom_label.bind("<Enter>", on_custom_enter)
        custom_label.bind("<Leave>", on_custom_leave)
        custom_label.bind("<Button-1>", lambda e: self.show_custom_response_dialog())
        
        self.response_buttons.append((custom_frame, custom_label))
        
        # Настраиваем растяжение колонок в responses_frame
        self.responses_frame.columnconfigure(0, weight=1)
        self.responses_frame.columnconfigure(1, weight=1)
        
    def select_response(self, selected_response):
        # Добавляем выбранный ответ в чат
        self.append_to_chat(selected_response, is_user=False)
        
        # Очищаем кнопки
        for frame, label in self.response_buttons:
            frame.destroy()
        self.response_buttons.clear()
        
        # Обучаем бота на выбранном ответе
        self.bot.learn_from_interaction(self.current_user_input, selected_response)
        
        # Сбрасываем состояние выбора
        self.choosing_response = False
        self.input_field.config(state='normal')
        
    def send_message(self, event=None):
        if self.choosing_response:
            return
            
        message = self.input_field.get().strip()
        if not message:
            return
            
        # Сохраняем текущий ввод для обучения
        self.current_user_input = message
        
        # Очищаем поле ввода
        self.input_field.delete(0, tk.END)
        
        # Добавляем сообщение пользователя в чат
        self.append_to_chat(message)
        
        # Генерируем варианты ответов
        self.current_responses = self.generate_response_options(message)
        
        # Показываем варианты ответов
        self.choosing_response = True
        self.input_field.config(state='disabled')
        self.show_response_options(self.current_responses)

def main():
    root = tk.Tk()
    app = ChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 