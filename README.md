- 👋 Hi, I’m @azad92KI
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
azad92KI/azad92KI is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Laden und Vorbereiten der Daten
# Nehmen Sie hier Ihre Grundriss-Skizze als Eingabedaten
input_image = Image.open('grundriss_skizze.png')
input_image = input_image.resize((256, 256))  # Größe anpassen, falls erforderlich
input_image = np.array(input_image) / 255.0  # Normalisierung

# Laden des trainierten Pix2Pix-Modells
model = tf.keras.models.load_model('pix2pix_model.h5')

# Generieren des Ausgabe-Bildes
output_image = model.predict(np.expand_dims(input_image, axis=0))[0]

# Anzeigen und Speichern des generierten Bildes
plt.imshow(output_image)
plt.axis('off')
plt.show()
plt.savefig('ausgabe_bild.png')
