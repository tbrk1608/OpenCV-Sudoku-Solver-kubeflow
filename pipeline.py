import kfp
from kfp import dsl


@dsl.pipeline(name="OSS Pipeline", description="")
def oss_pipeline(data_path):  # , model_path, save_path):

    load_process = kfp.components.load_component_from_file(
        "load_process_data\load_process.yaml"
    )
    cnn_model = kfp.components.load_component_from_file("cnn_model\cnn_model.yaml")
    flask_app = kfp.components.load_component_from_file("flask_app\/app.yaml")

    # Run tasks
    load_task = load_process(data_path)
    # cnn_model_task = cnn_model(load_task.outputs)
    # app_task = flask_app(cnn_model_task.outputs)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(oss_pipeline, "pipeline.yaml")
    # kfp.Client().create_run_from_pipeline_func(basic_pipeline, arguments={})
