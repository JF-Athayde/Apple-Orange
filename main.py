from anubia import *

choice = input('Do you want to train with new data? (Y or N): ')

X_train = [[0.85, 0.8, 0.9, 0.1], [0.25, 0.5, 0, 0.9]] # [texture, sweetness, red, yellow]
Y_train = [[0], [1]]

if choice == 'Y':
    num = int(input('How many entries do you want to include?: '))
    
    for i in range(num):
        print('Catalog fruit', i)
        X_train.append([float(input('Texture: ')), float(input('Sweetness: ')), float(input('Red: ')), float(input('Yellow: '))])
        Y_train.append(float('Fruit number: '))

model = DeepLearning(np.array(X_train), np.array(Y_train), learning_rate=0.01, hidden_layers=[5, 5], activation='sigmoid')
model.train(epochs=100000, verbose=True)


predictions = model.predict(X_train)

print('Error table')
for a, b in zip(predictions, Y_train):
    print(f'¦ {a[0]:.10f} ¦ {b[0]} ¦ {(1-abs(a[0]-b[0]))*100:.10f} %')

print('New predicts')
print('Predict:', model.predict([float(input('Texture: ')), float(input('Sweetness: ')), float(input('Red: ')), float(input('Yellow: '))]))
