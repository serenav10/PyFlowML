**PyFlowML**, currently in its early development stage as of May 2024, is a prototype developed within PyFlow's open-source, flow-based environment. **PyFlow**, a Python Qt visual scripting framework, is designed to facilitate GUI creation through visual programming, allowing non-technical users in computing to construct applications by dragging and dropping components onto a graphical canvas. Core elements of PyFlow include **nodes** (for operations and functions), **pins** (as connection ports), and **arcs** (to transmit data).

PyFlowML introduces **machine learning (ML)** analysis through four types of node categories: Data Load, Data Visualization, Data Classification, and Explainable AI (XAI). This integration leverages XAI techniques (e.g., SHAP) to clarify and demystify the decision-making processes in ML-based systems, thus enhancing understanding, trust, and trustworthiness among users.

Thus far, PyFlowML has been employed in research to investigate how Visual Programming Languages (VPLs) and no-code platforms can foster user participation in designing ML-based systems.

![quickdemo](gif/PyFlowML.gif)

## Installation

PyFlowML, like the PyFlowOpenCv package, is an extension of PyFlow and requires PyFlow to be installed beforehand. While it's not mandatory, it's advisable to install PyFlowOpenCv as well. Instructions for installing both PyFlow and PyFlowOpenCv can be found here: 

[Installation](https://pyflowopencv.readthedocs.io/en/latest/intro.html#installation)

After PyFlow installed through pip or setup.py. Clone or download PyFlowML repository to a local folder: 

```bash
git clone https://github.com/serenav10/PyFlowML.git
```

Go to the source code folder and install requirements:

```bash
cd PyFlowML
pip install -r requirements.txt
```

## Demo

Check out the [PyFlowML YouTube Demo](https://www.youtube.com/watch?v=N_8Q_R5lXrE).

## Publications

coming soon

## Author

[Serena Versino](https://github.com/serenav10)

## License
For more information on the licensing of PyFlowML and the conditions under which it is provided, please see the [license](LICENSE) file in this repository.

