import numpy as np
import matplotlib.pyplot as plt
from utils import *  # Utilities for loading data and running tests / Veriyi yüklemek ve testleri çalıştırmak için gerekli yardımcı fonksiyonlar
import copy
import math

def load_data():
    """
    Loads and returns the training data for the linear regression problem.
    Doğrusal regresyon problemi için eğitim verilerini yükler ve döndürür.
    
    Returns:
        x_train (ndarray): Input data, population of cities.
                           Girdi verisi, şehirlerin nüfusu.
        y_train (ndarray): Output data, profit in $10,000s.
                           Çıktı verisi, 10.000$ cinsinden kar.
    """
    # Example dataset / Örnek veri seti
    x_train = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598])
    y_train = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])
    
    return x_train, y_train

# Load the dataset / Veri setini yükle
x_train, y_train = load_data()

# Print x_train
# x_train verisini yazdır
print("Type of x_train:", type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

# Print y_train
# y_train verisini yazdır
print("Type of y_train:", type(y_train))
print("First five elements of y_train are:\n", y_train[:5])

# Print shapes of x_train and y_train and the number of training examples
# x_train ve y_train'in şekillerini ve eğitim örneklerinin sayısını yazdır
print('The shape of x_train is:', x_train.shape)
print('The shape of y_train is:', y_train.shape)
print('Number of training examples (m):', len(x_train))

# Create a scatter plot of the training data / Eğitim verisinin scatter plot'unu oluştur
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title and axis labels / Başlık ve eksen etiketlerini ayarla
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# Function to compute cost / Maliyet fonksiyonunu hesaplayan fonksiyon
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    Doğrusal regresyon için maliyet fonksiyonunu hesaplar.
    
    Args:
        x (ndarray): Input to the model (Population of cities) / Model girişi (Şehirlerin nüfusu)
        y (ndarray): Labels (Actual profits for the cities) / Etiket (Şehirlerin gerçek karları)
        w, b (scalar): Parameters of the model / Modelin parametreleri
    
    Returns:
        total_cost (float): The cost of using w,b as the parameters for linear regression / 
                            Doğrusal regresyon için maliyet
    """
    # Number of training examples / Eğitim örneklerinin sayısı
    m = x.shape[0]
    
    # Compute the cost / Maliyet fonksiyonunu hesapla
    total_cost = (1 / (2 * m)) * np.sum((w * x + b - y) ** 2)
    
    return total_cost

# Compute cost with initial parameters / Başlangıç parametreleriyle maliyeti hesapla
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Run public tests / Public testleri çalıştır
from public_tests import *
compute_cost_test(compute_cost)

# Function to compute gradient / Gradyan hesaplama fonksiyonu
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression.
    Doğrusal regresyon için gradyanı hesaplar.
    
    Args:
        x (ndarray): Input to the model (Population of cities) / Model girişi (Şehirlerin nüfusu)
        y (ndarray): Labels (Actual profits for the cities) / Etiket (Şehirlerin gerçek karları)
        w, b (scalar): Parameters of the model / Modelin parametreleri  
    
    Returns:
        dj_dw (scalar): The gradient of the cost w.r.t. the parameter w / w parametresine göre maliyetin gradyanı
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b / b parametresine göre maliyetin gradyanı     
    """
    # Number of training examples / Eğitim örneklerinin sayısı
    m = x.shape[0]
    
    # Compute the gradients / Gradyanı hesapla
    dj_dw = (1 / m) * np.sum((w * x + b - y) * x)
    dj_db = (1 / m) * np.sum(w * x + b - y)
    
    return dj_dw, dj_db

# Compute and print gradient with w and b initialized to zeros / w ve b'nin sıfır olduğu durumda gradyanı hesapla ve yazdır
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

# Run gradient tests / Gradyan testlerini çalıştır
compute_gradient_test(compute_gradient)

# Compute and print gradient with non-zero w and b / w ve b'nin sıfır olmadığı durumda gradyanı hesapla ve yazdır
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

# Gradient descent algorithm / Gradyan inişi algoritması
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn parameters.
    Gradyan inişi algoritmasını kullanarak parametreleri öğrenir.
    
    Args:
        x : (ndarray): Model input / Model girişi
        y : (ndarray): True labels / Gerçek etiketler
        w_in, b_in : (scalar) Initial values of model parameters / Modelin başlangıç parametre değerleri
        cost_function: Function to compute cost / Maliyet fonksiyonunu hesaplayan fonksiyon
        gradient_function: Function to compute the gradient / Gradyanı hesaplayan fonksiyon
        alpha : (float) Learning rate / Öğrenme hızı
        num_iters : (int) Number of iterations for gradient descent / Gradyan inişi algoritmasının kaç iterasyon çalışacağı
    
    Returns:
        w : (ndarray): Updated model parameters / Model parametrelerinin güncellenmiş değerleri
        b : (scalar) Updated model parameter / Model parametrelerinin güncellenmiş değeri
    """
    # Number of training examples / Eğitim örneklerinin sayısı
    m = len(x)
    
    # Lists to store cost and w history / Maliyet ve w geçmişi için listeler
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  # Avoid modifying global w within function / w'yi fonksiyon içinde değiştirmemek için kopyala
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters / Gradyanı hesapla ve parametreleri güncelle
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update parameters / Parametreleri güncelle
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration / Maliyeti her iterasyonda kaydet
        if i < 100000:  # Prevent resource exhaustion / Kaynak tüketimini önlemek için
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every 10% of the iterations / Maliyeti her 10 iterasyonda bir yazdır
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
        
    return w, b, J_history, w_history  # Return w and J, w history for graphing / w ve J, w geçmişini döndür

# Run gradient descent / Gradyan inişini çalıştır
initial_w = 0
initial_b = 0
learning_rate = 0.01
num_iterations = 1000

w, b, J_history, w_history = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, learning_rate, num_iterations)

# Compute predicted values and plot the graph / Tahmin edilen değerleri hesapla ve grafiği oluştur
predicted = w * x_train + b
plt.plot(x_train, predicted, c="b")

# Create a scatter plot of the training data / Eğitim verisinin scatter plot'unu oluştur
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title and axis labels / Başlık ve eksen etiketlerini ayarla
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')

plt.show()

# Compute and print profit predictions for specific population values / Belirli nüfus değerleri için kar tahminlerini hesapla ve yazdır
predict1 = 3.5 * w + b
print(f'For population = 35,000, we predict a profit of ${predict1*10000:.2f}')

predict2 = 7.0 * w + b
print(f'For population = 70')

