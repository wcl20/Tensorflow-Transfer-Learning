from tensorflow.keras.applications import VGG16

def main():

    model = VGG16(weights="imagenet", include_top=False)
    # Show layers
    for i, layer in enumerate(model.layers):
        print(f"[INFO] {i:03d}: {layer.__class__.__name__}")

if __name__ == '__main__':
    main()
