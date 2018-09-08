'''
Note that the img is processed in BGR space
'''
def run_test(IMG_SIZE = 128, N_IMG_MAX = 1000):
    from Source import StyleTransferNet
    from sklearn.model_selection import train_test_split
    import numpy as np
    import os
    import cv2
    from pathlib import Path


    # DEFINE PATHS:
    d = Path().resolve()
    data_path = str(d) + "/"
    predictions_path = str(d) + "/Predictions/"
    style_path = data_path + "la_muse.jpg"
    test_path = data_path + "MiniCOCO/" + str(IMG_SIZE) + "/"
    weight_load_path = str(d) + "/weights/model.ckpt"

    style_img = cv2.imread(style_path).astype(np.float32)

    X = []
    names = []
    model = StyleTransferNet(
        style_img,
        pretrained_path = None,
        is_training = False
    )

    n_img_processed = 0
    for filename in os.listdir(test_path):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        img = cv2.imread(test_path + filename)
        names.append(filename)
        X.append(img)
        n_img_processed += 1

        if n_img_processed == N_IMG_MAX:
            break

    X = np.array(X)
    names = np.array(names)


    model.load_weights(weight_load_path)
    predictions = model.predict(X)

    for ind in range(X.shape[0]):
        filename = predictions_path + names[ind][:-4] + "_styled.png"
        original_file = predictions_path + names[ind]

        cv2.imwrite(filename, predictions[ind])
        cv2.imwrite(original_file, X[ind])

# run(1, N_IMG_MAX = 10)
run_test(N_IMG_MAX = 100)