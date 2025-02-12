class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(project=args.project, config=args)

    def log_epoch_metrics(self, metrics):
        """
        Log train/test metrics onto W&B.
        """

        # Log number of model parameters as W&B summary
        self._wandb.summary["n_parameters"] = metrics.get("n_parameters", None)
        metrics.pop("n_parameters", None)

        # Log current epoch
        self._wandb.log({"epoch": metrics.get("epoch")}, commit=False)
        metrics.pop("epoch")

        for k, v in metrics.items():
            if "train" in k:
                self._wandb.log({f"Global Train/{k}": v}, commit=False)
            elif "test" in k:
                self._wandb.log({f"Global Test/{k}": v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric(
            "Rank-0 Batch Wise/*", step_metric="Rank-0 Batch Wise/global_train_step"
        )

        # Set epoch-wise step
        self._wandb.define_metric("Global Train/*", step_metric="epoch")
        self._wandb.define_metric("Global Test/*", step_metric="epoch")
