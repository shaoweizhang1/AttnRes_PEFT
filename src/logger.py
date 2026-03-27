import os


class Logger:
    def __init__(self, args):
        self.args = args
        self.run = None

    def setup_wandb(self):
        if not self.args.use_wandb:
            return None

        import wandb

        if self.args.wandb_project:
            os.environ["WANDB_PROJECT"] = self.args.wandb_project
        if self.args.wandb_run_name:
            os.environ["WANDB_NAME"] = self.args.wandb_run_name
        if self.args.wandb_entity:
            os.environ["WANDB_ENTITY"] = self.args.wandb_entity

        self.run = wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            entity=self.args.wandb_entity,
            config=vars(self.args),
        )
        return self.run

    def finish(self):
        if self.run is not None:
            self.run.finish()
