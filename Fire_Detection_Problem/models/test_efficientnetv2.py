from efficientnetv2 import build_model


def main():
    model = build_model(input_shape=(224, 224, 3), variant='b0', fine_tune=False)
    model.summary()


if __name__ == '__main__':
    main()
