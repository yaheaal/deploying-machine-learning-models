from pipeline import main_pipeline
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset, save_pipeline

config = config


def run_training() -> None:
    """
    Train the Model
    """

    data = load_dataset(file_name=config.app.data_file)

    X_df, y_df = data.drop(config.model.target, axis=1), data[config.model.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_df,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
    )

    main_pipeline.fit(X_train, y_train)

    save_pipeline(pipeline_to_presist=main_pipeline)


if __name__ == "__main__":
    run_training()
