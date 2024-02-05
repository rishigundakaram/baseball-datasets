import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

def load_data(path='/home/projects/baseball-datasets/data/final/atbats.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[['date', 'pitcher', 'batter', 'outcome']]
    train_df, valid_df, test_df = split_data(df)
    return train_df, valid_df, test_df


def split_data(df, train_cuttoff='2022-06-01', test_cuttoff='2023-01-01'):
    train = df[df['date'] < train_cuttoff]
    valid = df[(df['date'] >= train_cuttoff) & (df['date'] < test_cuttoff)]
    test = df[df['date'] >= test_cuttoff]
    return train, valid, test

class DataPreprocessor:
    def __init__(self, train_df, proportion=.15):
        self.train_df = train_df
        self.known_batter_ids = set(train_df['batter'])
        self.known_pitcher_ids = set(train_df['pitcher'])

        # Create mappings for batter and pitcher IDs to indices
        self.batter_id_to_index = {id: idx for idx, id in enumerate(self.known_batter_ids, start=1)}  # Start from 1 to reserve 0 for 'unknown'
        self.pitcher_id_to_index = {id: idx for idx, id in enumerate(self.known_pitcher_ids, start=1)}  # Start from 1 to reserve 0 for 'unknown' 
        self.batter_id_to_index['unknown'] = 0
        self.pitcher_id_to_index['unknown'] = 0
        self.outcomes = {
            'single': 0, 
            'double': 1,
            'triple': 2,
            'homerun': 3,
            'walk': 4,
            'strikeout': 5,
            'groundout': 6,
            'flyout': 7,
            'lineout':8,
            'popout': 9,
            'unknown': 10
            }
    
    def augment(self, train_df, proportion=.15):
        # sample the data according to the proportion and remove either the pitcher or batter
        train_aug_pitcher = train_df.sample(frac=proportion/2)
        train_aug_pitcher['pitcher'] = 'unknown'
        train_aug_batter = train_df.sample(frac=proportion/2)
        train_aug_batter['batter'] = 'unknown'
        train_aug = pd.concat([train_df, train_aug_pitcher, train_aug_batter])
        return train_aug

    def preprocess(self, df, train=False, proportion=.15):
        # Replace unknown IDs and convert to indices
        processed_df = df.copy()
        if train:
            processed_df = self.augment(df, proportion=proportion)

        
        processed_df.loc[:, 'batter'] = processed_df['batter'].apply(lambda x: self.batter_id_to_index.get(x, 0))
        processed_df.loc[:, 'pitcher'] = processed_df['pitcher'].apply(lambda x: self.pitcher_id_to_index.get(x, 0))
        processed_df.loc[:, 'outcome'] = processed_df['outcome'].apply(lambda x: self.outcomes.get(x, 10))
        # Also need to drop the date column
        processed_df = processed_df.drop(columns=['date'])

        return processed_df

    def get_num_batters(self):
        return len(self.batter_id_to_index)
    
    def get_num_pitchers(self):
        return len(self.pitcher_id_to_index)    
    
    def get_num_outcomes(self):
        return len(self.outcomes)

class BaseballDataset(Dataset):
    def __init__(self, df):
        self.pitcher = torch.tensor(df['pitcher'].to_numpy().astype(int), dtype=torch.long)
        self.batter = torch.tensor(df['batter'].to_numpy().astype(int), dtype=torch.long)
        self.data = torch.tensor(df['outcome'].to_numpy().astype(int), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the first two columns are batter and pitcher indices, and the rest are outcomes
        return self.pitcher[idx], self.batter[idx], self.data[idx]



class NCF(nn.Module):
    def __init__(
        self, num_pitchers, num_batters, num_outcomes, embed_dim=50, num_layers=2
    ):
        super(NCF, self).__init__()

        # Increase the embedding dimension
        self.pitcher_embedding = nn.Embedding(num_pitchers, embed_dim)
        self.batter_embedding = nn.Embedding(num_batters, embed_dim)

        self.fc1 = nn.Linear(embed_dim * 2, 128)  # Increase the hidden layer size
        self.linear_layers = nn.ModuleList(
            [nn.Linear(128, 128) for _ in range(num_layers - 2)]
        )
        self.dropout = nn.ModuleList([nn.Dropout(0.3) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(128, num_outcomes)

        # Add Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, pitcher_ids, batter_ids):
        pitcher_embed = self.pitcher_embedding(pitcher_ids)
        batter_embed = self.batter_embedding(batter_ids)
        x = torch.cat([pitcher_embed, batter_embed], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout[0](x)  # Dropout layer after first hidden layer
        for idx in range(len(self.linear_layers)):
            x = F.relu(self.linear_layers[idx](x))
            x = self.dropout[idx + 1](x)
        
        out = self.output_layer(x)
        
        return out


    
def main(): 
    # wandb.init(
    #     project="baseball-simulation",
    #     config={
    #         "epochs": 30,
    #         "batch_size": 1024,
    #         'num_layers': 3, 
    #         "embed_dim": 50,
    #         "learning_rate": 0.001,
    #         'early_stop_patience': 3,
    #         "architecture": "NCF-cat",
    #     },
    #     entity='rishi-gundakaram'
    # )
    run = wandb.init()

    LEARNING_RATE = wandb.config.learning_rate
    EPOCHS = wandb.config.epochs
    EMBED_DIM = wandb.config.embed_dim
    NUM_LAYERS = wandb.config.num_layers
    EARLY_STOP_PATIENCE =  wandb.config.early_stop_patience
    BATCH_SIZE = wandb.config.batch_size

    train_df, valid_df, test_df = load_data()
    preprocessor = DataPreprocessor(train_df)

    train_df_processed = preprocessor.preprocess(train_df, train=True)
    valid_df_processed = preprocessor.preprocess(valid_df)
    test_df_processed = preprocessor.preprocess(test_df)

    # Create Dataset Instances
    train_dataset = BaseballDataset(train_df_processed)
    valid_dataset = BaseballDataset(valid_df_processed)
    test_dataset = BaseballDataset(test_df_processed)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_pitchers = preprocessor.get_num_pitchers()
    num_batters = preprocessor.get_num_batters()
    num_outcomes = preprocessor.get_num_outcomes()
    
    model = NCF(num_pitchers, num_batters, num_outcomes, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using: ', device)
    # device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping variables
    prev_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # Training loop
        model.train()
        train_loss = 0.0
        for i, (pitchers, batters, outcomes) in enumerate(train_dataloader):
            pitchers, batters, outcomes = pitchers.to(device), batters.to(device), outcomes.to(device)
            # check all data types
            outputs = model(pitchers, batters)
            loss = criterion(input=outputs, target=outcomes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pitchers, batters, outcomes in valid_dataloader:
                pitchers, batters, outcomes = pitchers.to(device), batters.to(device), outcomes.to(device)
                outputs = model(pitchers, batters)
                loss = criterion(outputs, outcomes)
                val_loss += loss.item()
        val_loss /= len(valid_dataloader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # Early stopping check
        if val_loss > prev_val_loss:
            epochs_no_improve += 1
            if epochs_no_improve == EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            epochs_no_improve = 0
        prev_val_loss = val_loss

    return val_loss

main()