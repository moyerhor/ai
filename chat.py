from titan_llm import ChatBot
import os

def main():
    checkpoint_dir = 'checkpoints'
    model_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_1.pth')
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели {model_path} не найден!")
        print("Сначала нужно обучить модель, запустив trainer.py")
        return
        
    chatbot = ChatBot(model_path=model_path)
    
    print("\nЧат-бот готов к работе! Введите 'выход' для завершения.")
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
            
        response = chatbot.generate_response(user_input)
        print(f"Бот: {response}")

if __name__ == "__main__":
    main() 