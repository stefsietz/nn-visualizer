
# Interactive 3D Neural Network Visualizer

This project serves as a tool to make the structure of neural networks more visually clear than with usual schematic representations and provides interactive ways to edit architectural parameters. It can be used for teaching beginners how the layers of a deep neural network or CNN are connected between each other and visualizes the number of weights / calculations involved in the model.

There are two "modes":
1. Constructing the CNN in the Unity Editor by combining the prefabs and adjusting parameters.
2. Loading a  JSON file with weights and activations from an actual tensorflow model to be displayed, which has to be set up and converted first.

## Getting Started

These instructions should get you up and running, if there are any issues don't hesitate to ask me.

### Prerequisites
Until now this project has exclusively been developed on Windows and I haven't tested it on my Linux system yet.
For the tensorflow scripts you obviously need tensorflow and numpy installed.

### Installing

Install the latest Version of Unity.
The CNN Editor should work right away with the included scene (in the Assets/Scenes directory), just drag the layer prefabs into the object manager (don't drag them into the scene as they all should be positioned at the origin)

For the model loading, please have a look at the "cifar10" python project. It includes code for building and training a simple cifar10 classification CNN with the necessary layer naming and checkpoint output, as well as the ckpt converter, that runs cifar10 test examples to get the according neuron activations and writes out a JSON that can then be loaded by the Unity project.

Alternatively load this set of example JSONs: https://drive.google.com/open?id=1Hr3sveB1hiVO37OmwOWrOLU8EqyMmrG3
You have to select the containing folder after pressing the "Load" button.

## Contributing

Contributions are very welcome! I think a lot of layer types still need to be implemented, also parallel architectures like for example used in Faster RCNN are not possible at the moment, and a lot of refactoring has to be done to make the project more extendable.


## Authors

* **Stefan Sietzen** - *Initial work* - [visuality.at](http://visuality.at)

Add yourself  after contributing ;-)

## License

This project is licensed under the Apache License 2.0

## Acknowledgments

* Thanks to Manuela Waldner of TU Vienna for initial discussions about this project, which originated as a course excercise for the Visualization 2 course and for allowing the execution of own ideas instead of existing paper implementations for the excercise.
