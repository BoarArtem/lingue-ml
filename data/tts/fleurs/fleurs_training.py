from models.tacotron2 import *
from data.tts.fleurs.fleurs_dataset_executor import extract_dict_of_loaders

loss_fn = get_tacotron2_loss()
model = load_tacotron2(f"{PROJECT_ROOT}/tacotron2_final.pth")

MAIN_DIRECTORY = f"{PROJECT_ROOT}/data/tts/fleurs"

def train_tacotron2(train_key="train", test_key="test", batch_size=32):
    dataloaders = extract_dict_of_loaders(MAIN_DIRECTORY, train_key, test_key, batch_size)

    print(f"Dataloaders extracted; Length: {len(dataloaders)}")

    for dataloader_name, dataloader_value in dataloaders.items():
        print(f"Training on {dataloader_name} dataset")
        train_test(model, dataloader_value[train_key], 100, loss_fn, get_optimizer(model, 1e-4), test_dataloader=dataloader_value[test_key])

    torch.save(model.state_dict(), f"{PROJECT_ROOT}/tacotron2_final.pth")

if __name__ == '__main__':
    train_tacotron2()