import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pyfirmata import Arduino, util
import json
import os


class AntennaTracker:
    def __init__(self, port='/dev/ttyACM0', emulate=False):
        self.emulate_mode = emulate
        self.signal_lost = False
        self.last_update_time = time.time()

        self.config = {
            'SIGMOID_SLOPE': 1.0,
            'SIGMOID_OFFSET': 4.0,
            'RSSI_OFFSETS': [0, 0, 0, 0],  # left, right, up, down
            'SERVO_LIMITS': [0, 180],
            'MAX_STEP': 2.0,
            'MIN_STEP': 0.05,
            'DEADBAND': 3,
            'RSSI_MIN': 120,
            'RSSI_MAX': 400
        }

        self.load_config()

        if not self.emulate_mode:
            try:
                self.board = Arduino(port)
                self.servo_pan = self.board.get_pin(f'd:5:s')
                self.servo_tilt = self.board.get_pin(f'd:6:s')
                print(f"Успешное подключение к Arduino на порту {port}")
            except Exception as e:
                print(f"Ошибка подключения: {e}")
                self.emulate_mode = True
                print("Активирован режим эмуляции")

        self.angle_pan = 90
        self.angle_tilt = 90
        self.history = deque(maxlen=100)
        self.rssi_history = deque(maxlen=100)

        # Калибровка
        self.calibration_data = {
            'min_rssi': self.config['RSSI_MIN'],
            'max_rssi': self.config['RSSI_MAX'],
            'offsets': self.config['RSSI_OFFSETS']
        }

        # Визуализация
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(10, 8))
        self.setup_plots()

        # Переход в начальное положение
        self.move_to(90, 90)

    def setup_plots(self):
        """Настройка графиков для визуализации"""
        self.pan_line, = self.ax1.plot([], [], 'b-', label='Pan')
        self.tilt_line, = self.ax1.plot([], [], 'r-', label='Tilt')
        self.rssi_lines = []
        colors = ['g-', 'm-', 'c-', 'y-']
        for i, color in enumerate(colors):
            line, = self.ax2.plot([], [], color, label=f'RSSI {i}')
            self.rssi_lines.append(line)

        self.ax1.set_title('Углы сервоприводов')
        self.ax1.set_ylim(0, 180)
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.set_title('Уровни RSSI')
        self.ax2.set_ylim(self.config['RSSI_MIN'] - 20, self.config['RSSI_MAX'] + 20)
        self.ax2.legend()
        self.ax2.grid(True)

        plt.tight_layout()

    def update_plots(self):
        """Обновление графиков"""
        if len(self.history) == 0:
            return

        x = range(len(self.history))
        pan_data = [h[0] for h in self.history]
        tilt_data = [h[1] for h in self.history]

        self.pan_line.set_data(x, pan_data)
        self.tilt_line.set_data(x, tilt_data)
        self.ax1.relim()
        self.ax1.autoscale_view()

        if len(self.rssi_history) > 0:
            for i in range(4):
                rssi_data = [r[i] for r in self.rssi_history]
                self.rssi_lines[i].set_data(range(len(rssi_data)), rssi_data)
            self.ax2.relim()
            self.ax2.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    def load_config(self, filename='antenna_config.json'):
        """Загрузка конфигурации из файла"""
        if os.path.exists(filename):
            try:
                with open(filename) as f:
                    saved_config = json.load(f)
                    for key in self.config:
                        if key in saved_config:
                            self.config[key] = saved_config[key]
                print("Конфигурация загружена")
            except Exception as e:
                print(f"Ошибка загрузки конфига: {e}")

    def save_config(self, filename='antenna_config.json'):
        """Сохранение конфигурации в файл"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=4)
            print("Конфигурация сохранена")
        except Exception as e:
            print(f"Ошибка сохранения конфига: {e}")

    def calibrate(self, duration=10):
        """Автоматическая калибровка RSSI"""
        print("Начало калибровки...")
        samples = []
        start_time = time.time()

        while time.time() - start_time < duration:
            rssi = self.read_rssi()
            samples.append(rssi)
            time.sleep(0.1)
            print(".", end='', flush=True)

        print("\nКалибровка завершена!")

        all_rssi = np.array(samples)
        self.calibration_data['min_rssi'] = int(np.percentile(all_rssi.flatten(), 5))
        self.calibration_data['max_rssi'] = int(np.percentile(all_rssi.flatten(), 95))

        means = np.mean(all_rssi, axis=0)
        base = np.max(means)
        self.calibration_data['offsets'] = [int(base - m) for m in means]

        self.config['RSSI_MIN'] = self.calibration_data['min_rssi']
        self.config['RSSI_MAX'] = self.calibration_data['max_rssi']
        self.config['RSSI_OFFSETS'] = self.calibration_data['offsets']

        print(f"Min RSSI: {self.config['RSSI_MIN']}")
        print(f"Max RSSI: {self.config['RSSI_MAX']}")
        print(f"Offsets: {self.config['RSSI_OFFSETS']}")

        self.save_config()

    def read_rssi(self):
        """Чтение RSSI с датчиков"""
        if self.emulate_mode:
            t = time.time()
            noise = np.random.normal(0, 5, 4)
            base = np.array([
                100 * np.sin(t * 0.5) + 200,
                100 * np.cos(t * 0.5) + 200,
                80 * np.sin(t * 0.3) + 180,
                80 * np.cos(t * 0.3) + 180
            ])
            return np.clip(base + noise, self.config['RSSI_MIN'], self.config['RSSI_MAX'])
        else:
            try:
                return np.array([
                    self.board.analog[0].read() or 0,
                    self.board.analog[1].read() or 0,
                    self.board.analog[2].read() or 0,
                    self.board.analog[3].read() or 0
                ])
            except:
                return np.zeros(4)

    def process_rssi(self, raw_rssi):
        """Обработка RSSI с учетом калибровки"""
        adjusted = raw_rssi + np.array(self.config['RSSI_OFFSETS'])

        return np.clip(adjusted, self.config['RSSI_MIN'], self.config['RSSI_MAX'])

    def check_signal_loss(self, rssi_values):
        """Проверка потери сигнала"""
        if np.all(rssi_values < self.config['RSSI_MIN'] + 10):
            if not self.signal_lost:
                print("ПРЕДУПРЕЖДЕНИЕ: Сигнал потерян!")
                self.signal_lost = True
            return True
        else:
            if self.signal_lost:
                print("Сигнал восстановлен")
                self.signal_lost = False
            return False

    def calculate_angles(self, rssi_values):
        """Вычисление новых углов на основе RSSI"""
        if self.signal_lost:
            return None, None  # Не двигаемся при потере сигнала

        right, left = rssi_values[1], rssi_values[0]
        diff_h = right - left
        deadband_h = self.config['DEADBAND'] * (np.mean([right, left]) - self.config['RSSI_MIN']) / (
                self.config['RSSI_MAX'] - self.config['RSSI_MIN'])

        if abs(diff_h) > deadband_h:
            x = diff_h / 10.0
            pan_step = self.config['MAX_STEP'] / (
                    1 + np.exp(-self.config['SIGMOID_SLOPE'] * x + self.config['SIGMOID_OFFSET']))
            pan_step = np.sign(diff_h) * min(abs(pan_step), self.config['MAX_STEP'])
            if abs(pan_step) < self.config['MIN_STEP']:
                pan_step = 0
        else:
            pan_step = 0

        up, down = rssi_values[2], rssi_values[3]
        diff_v = up - down
        deadband_v = self.config['DEADBAND'] * (np.mean([up, down]) - self.config['RSSI_MIN']) / (
                self.config['RSSI_MAX'] - self.config['RSSI_MIN'])

        if abs(diff_v) > deadband_v:
            y = diff_v / 10.0
            tilt_step = self.config['MAX_STEP'] / (
                    1 + np.exp(-self.config['SIGMOID_SLOPE'] * y + self.config['SIGMOID_OFFSET']))
            tilt_step = np.sign(diff_v) * min(abs(tilt_step), self.config['MAX_STEP'])
            if abs(tilt_step) < self.config['MIN_STEP']:
                tilt_step = 0
        else:
            tilt_step = 0

        return pan_step, tilt_step

    def move_to(self, pan_angle, tilt_angle):
        """Плавное перемещение в указанную позицию"""
        self.angle_pan = max(self.config['SERVO_LIMITS'][0], min(self.config['SERVO_LIMITS'][1], pan_angle))
        self.angle_tilt = max(self.config['SERVO_LIMITS'][0], min(self.config['SERVO_LIMITS'][1], tilt_angle))

        if not self.emulate_mode:
            try:
                self.servo_pan.write(self.angle_pan)
                self.servo_tilt.write(self.angle_tilt)
            except:
                pass

        self.history.append((self.angle_pan, self.angle_tilt))
        self.update_plots()

    def run(self, update_interval=0.05):
        try:
            print("Запуск трекера...")
            while True:
                start_time = time.time()

                raw_rssi = self.read_rssi()
                processed_rssi = self.process_rssi(raw_rssi)
                self.rssi_history.append(processed_rssi)

                self.check_signal_loss(processed_rssi)

                pan_step, tilt_step = self.calculate_angles(processed_rssi)

                if pan_step is not None and tilt_step is not None:
                    new_pan = self.angle_pan + pan_step
                    new_tilt = self.angle_tilt + tilt_step
                    self.move_to(new_pan, new_tilt)

                if time.time() - self.last_update_time > 1.0:
                    print(f"Pan: {self.angle_pan:.1f}° | Tilt: {self.angle_tilt:.1f}° | RSSI: {processed_rssi}")
                    self.last_update_time = time.time()

                elapsed = time.time() - start_time
                if elapsed < update_interval:
                    time.sleep(update_interval - elapsed)

        except KeyboardInterrupt:
            print("\nОстановка трекера...")
            self.move_to(90, 90)
            if not self.emulate_mode:
                self.board.exit()
            plt.close()

# Перед началом работы нужно импортировать библиотеки командой в терминале pip install название библиотеки либо вручную в настройках редактора

# Если не нужен вывод текста в консоли - все команды с print можно удалить

# КОМАНДЫ:

# Для реальной работы:
# tracker = AntennaTracker(port='/dev/ttyACM0')

# Для эмуляции:
#  tracker = AntennaTracker(emulate=True)

# Калибровка:
# tracker.calibrate(duration=5)

# Запуск трекера:
# tracker.run()

# Для ручного управления сервами: tracker.move_to(pan, tilt)
# Для чтения RSSI: print(tracker.read_rssi())