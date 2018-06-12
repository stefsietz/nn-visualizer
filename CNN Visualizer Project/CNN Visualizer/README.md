Standard Geometry Shader Example
================================

![gif](https://i.imgur.com/hGtXkA7.gif)

This is an example that shows how to implement a geometry shader that is
compatible with the standard lighting model in Unity.

Writing geometry shaders is hard
--------------------------------

Implementing a geometry shader in Unity is not easy as it seems because
[surface shaders] don't allow geometry stage customization -- This means that
you have to implement the whole lighting passes by yourself without the help of
surface shaders.

This example shows the minimum implementation of vertex/geometry/fragment
shader set that provides a custom geometry modification along with the standard
lighting features.

[surface shaders]: https://docs.unity3d.com/Manual/SL-SurfaceShaders.html

Limitations
-----------

To make the example as simple as possible, some features are intensionally
omitted from the shader.

- No forward rendering support (!)
- No lightmap support
- No shadowmask support
- No motion vectors support
- No GPU instancing support
- It hasn't been tested with XR.

It's not impossible to add these features to the shader, that might be pretty
troublesome though.

License
-------

Copyright (c) 2017 Unity Technologies

This repository is to be treated as an example content of Unity; you can use
the code freely in your projects. Also see the [FAQ] about example contents.

[FAQ]: https://unity3d.com/unity/faq#faq-37863
