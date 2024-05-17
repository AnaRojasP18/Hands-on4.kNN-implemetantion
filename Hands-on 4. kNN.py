class KNN:
    def __init__(self, k=3):
        self.k = k

#El metodo fit recibe datos de entrenamiento...
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

#Calcula la distancia euclidiana entre 2 instancias...
    def euclidean_distance(self, instance1, instance2):
        distance = 0
        for i in range(len(instance1)):
            distance += (instance1[i] - instance2[i]) ** 2
        return distance ** 0.5

#Predice las etiquetas para los datos de prueba...
    def predict(self, X_test):
        predictions = []
        for instance in X_test:
            neighbors = self.get_neighbors(instance)
            output_values = [self.y_train[neighbor] for neighbor in neighbors]
            prediction = max(set(output_values), key=output_values.count)
            predictions.append(prediction)
        return predictions

#Encuentra los indices de los k vecinos más cercanos a una instancia de prueba...
    def get_neighbors(self, test_instance):
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(test_instance, self.X_train[i])
            distances.append((i, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = [dist[0] for dist in distances[:self.k]]
        return neighbors

# Ejemplo de uso:

#Datos de entrenamiento
X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
y_train = [0, 0, 1, 1, 1, 0, 1, 0]
#Datos de prueba
X_test = [[5, 6], [6, 7], [2, 2], [8, 9], [3, 5], [1, 8]]

knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(predictions)  # Output: [1, 0, 0, 0, 1, 1]
