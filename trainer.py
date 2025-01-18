import torch
from torch.utils.data import DataLoader, Dataset
from titan_llm import TitanMAC
from tqdm import tqdm
import time
import gc
import os

class TextDataset(Dataset):
    def __init__(self, file_path, sequence_length=64, vocab_size=256):
        # Читаем текстовый файл
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Преобразуем текст в токены и ограничиваем их значения
        self.tokens = [min(ord(c) % vocab_size, vocab_size-1) for c in text]
        self.sequence_length = sequence_length
        
    def __len__(self):
        return max(0, len(self.tokens) - self.sequence_length)
    
    def __getitem__(self, idx):
        # Получаем последовательность токенов
        input_tokens = self.tokens[idx:idx + self.sequence_length]
        target_tokens = self.tokens[idx + 1:idx + self.sequence_length + 1]
        
        # Преобразуем в тензоры
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        
        return input_tensor, target_tensor

class TitanTrainer:
    def __init__(self, model, learning_rate=1e-4, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.checkpoint_dir = checkpoint_dir
        
        # Создаем директорию для чекпоинтов если её нет
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
        
    def train_step(self, input_batch, target_batch):
        self.optimizer.zero_grad()
        
        # Получаем предсказания модели
        output = self.model(input_batch)
        
        # Reshape для функции потерь
        B, T, C = output.shape
        output = output.view(B * T, C)
        target_batch = target_batch.view(B * T)
        
        # Считаем потери
        loss = self.loss_fn(output, target_batch)
        
        # Обратное распространение
        loss.backward()
        self.optimizer.step()
        
        # Очищаем память
        del output
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return loss.item()
    
    def train(self, dataloader, epochs, start_epoch=0):
        self.model.train()
        print("\nНачинаем обучение модели:")
        print(f"Всего эпох: {epochs}")
        print(f"Размер датасета: {len(dataloader.dataset)} последовательностей")
        print(f"Размер батча: {dataloader.batch_size}")
        print(f"Шагов за эпоху: {len(dataloader)}")
        print(f"Используется {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        
        try:
            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                total_loss = 0
                
                progress_bar = tqdm(dataloader, desc=f'Эпоха {epoch+1}/{epochs}')
                
                for batch_idx, (input_batch, target_batch) in enumerate(progress_bar):
                    loss = self.train_step(input_batch, target_batch)
                    total_loss += loss
                    
                    progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                    
                    # Очищаем память каждые 10 батчей
                    if batch_idx % 10 == 0:
                        gc.collect()
                
                epoch_time = time.time() - epoch_start_time
                avg_loss = total_loss / len(dataloader)
                
                print(f"\nСтатистика эпохи {epoch+1}:")
                print(f"├── Средняя потеря: {avg_loss:.4f}")
                print(f"├── Время эпохи: {epoch_time:.2f} сек")
                print(f"└── Скорость: {len(dataloader.dataset) / epoch_time:.1f} последовательностей/сек")
                
                # Сохраняем чекпоинт
                self.save_checkpoint(epoch + 1, avg_loss)
                print(f"Сохранен чекпоинт epoch_{epoch+1}.pth\n")
                
        except KeyboardInterrupt:
            print("\nОбучение прервано пользователем. Последний чекпоинт сохранен.")
        
        except Exception as e:
            print(f"\nПроизошла ошибка: {str(e)}")
            print("Последний чекпоинт сохранен.")
            raise e

def main():
    # Уменьшенные параметры модели для слабого компьютера
    vocab_size = 256
    hidden_size = 256
    sequence_length = 64
    batch_size = 16
    epochs = 1
    
    # Создаём модель
    model = TitanMAC(vocab_size=vocab_size, hidden_size=hidden_size)
    
    # Создаём тренер
    trainer = TitanTrainer(model)
    
    # Проверяем наличие сохраненного чекпоинта
    checkpoint_dir = 'checkpoints'
    start_epoch = 0
    
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')])
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Найден чекпоинт: {checkpoints[-1]}")
            start_epoch, _ = trainer.load_checkpoint(latest_checkpoint)
            print(f"Продолжаем обучение с эпохи {start_epoch}")
    
    # Загружаем данные
    dataset = TextDataset('Текстовый документ.txt', sequence_length=sequence_length, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Запускаем обучение
    trainer.train(dataloader, epochs, start_epoch)

if __name__ == "__main__":
    main() 