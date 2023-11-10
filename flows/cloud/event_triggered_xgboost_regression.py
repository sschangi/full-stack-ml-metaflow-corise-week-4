from metaflow import FlowSpec, step, card, conda_base, project, current, Parameter, Flow, trigger
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@trigger(events=["s3"])
@conda_base(libraries={"conda-forge::xgboost": '1.5.1', "conda-forge::scikit-learn": '1.1.2', "conda-forge::pandas": '1.4.2', 
"conda-forge::pyarrow": '14.0.1.'})
@project(name="taxi_fare_prediction")
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        # TODO:
        # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
        # Understand what is happening.
        # Revisit task 1 and think about what might go in this function.
        obviously_bad_data_filters = [
        df.fare_amount > 0,  # fare_amount in US Dollars
        df.trip_distance <= 100,  # trip_distance in miles
        df.trip_distance > 0,
        # TODO: add some logic to filter out what you decide is bad data!
        # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.
        df.airport_fee > 0,
        df.congestion_surcharge > 0,
        df.total_amount > 0,
        df.tolls_amount > 0,
        df.tip_amount > 0,
        df.mta_tax > 0,
        df.extra > 0
        ]

        for f in obviously_bad_data_filters:
            df = df[f]

        df.dropna()

        Q1 = df['trip_distance'].quantile(0.25)
        Q3 = df['trip_distance'].quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the DataFrame to keep rows within the IQR
        df = df[(df['trip_distance'] >= lower_bound) & (df['trip_distance'] <= upper_bound)]
        
        return df

    @step
    def start(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        self.df = self.transform_features(pd.read_parquet(self.data_url))

        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
        # This is a simple/naive way to do this, and is meant to keep this example simple, to focus learning on deploying Metaflow flows.
        # In practice, you want split time series data in more sophisticated ways and run backtests.
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.xgboost_model)

    @step
    def xgboost_model(self):
        "Fit a single variable, xgboost model to the data."
        import xgboost
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedKFold
        from xgboost import XGBRegressor

        # TODO: Play around with the model if you are feeling it.
        self.model = XGBRegressor()

        self.next(self.validate)

    def gather_sibling_flow_run_results(self):
        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅"
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [
                    Markdown(icon),
                    Artifact(run.id),
                    Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                    Artifact(score),
                    Markdown(msg),
                ]
                rows.append(row)
            else:
                rows.append(
                    [
                        Markdown("✅"),
                        Artifact(run.id),
                        Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                        Artifact(str(self.scores.mean())),
                        Markdown("This run..."),
                    ]
                )
        return rows

    @card(type="corise")
    @step
    def validate(self):
        from numpy import absolute
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedKFold

        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        self.scores = cross_val_score(self.model, self.X, self.y, scoring='r2', cv=cv, n_jobs=-1)
        
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(
            Table(
                self.gather_sibling_flow_run_results(),
                headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"],
            )
        )
        self.next(self.end)

    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()
