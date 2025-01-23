import matplotlib.pyplot as plt
import numpy as np

#antal pads 
models = ['valid padding', 'same padding', 'over padding']

# Mean for modeller
means_imagenette = [0.6198726296424866, 0.6769426465034485, 0.6736305952072144]
means_mnist = [0.9391999840736389 , 0.9559000134468079 , 0.9545999765396118]

# CI upper og lower grænse
upper_bounds_imagenette = np.array(means_imagenette) + 1.96 * np.sqrt((np.array(means_imagenette)* (1 - np.array(means_imagenette)))/3925)
lower_bounds_imagenette = np.array(means_imagenette) - 1.96 * np.sqrt((np.array(means_imagenette)* (1 - np.array(means_imagenette)))/3925)

upper_bounds_mnist = np.array(means_mnist) + 1.96 * np.sqrt((np.array(means_mnist)* (1 - np.array(means_mnist)))/10000)
lower_bounds_mnist = np.array(means_mnist) - 1.96 * np.sqrt((np.array(means_mnist)* (1 - np.array(means_mnist)))/10000)

# Beregn fejllinjer (afstanden til middelværdien)
errors_imagenette = np.array([means_imagenette - np.array(lower_bounds_imagenette), np.array(upper_bounds_imagenette) - means_imagenette])
errors_mnist = np.array([means_mnist - np.array(lower_bounds_mnist), np.array(upper_bounds_mnist) - means_mnist])


print(upper_bounds_imagenette)
print(lower_bounds_imagenette)
print(upper_bounds_mnist)
print(lower_bounds_mnist)



# Plot CI
plt.figure(figsize=(8, 5))
plt.errorbar(models, means_imagenette, yerr=errors_imagenette, fmt='o', markersize = 3, capsize=5, color='orange', label='95% CI Imagenette')
plt.errorbar(models, means_mnist, yerr=errors_mnist, markersize = 3,fmt='o', capsize=5, color='red', label='95% CI MNIST')
plt.ylabel('Accuracy')
plt.xlabel('padding type')
plt.title('Konfidensinterval for modeller')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()




