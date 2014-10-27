import cStringIO, os, struct, sys
import random
from OpenGL.GL.shaders import compileProgram, compileShader
import PIL.Image
from ctypes import *
from pyglet.gl import *

SCREEN_W = 512
SCREEN_H = 512


class HeatWindow(pyglet.window.Window):
    width = None
    height = None
    fbo = None
    texture = None
    palette = None

    def __init__(self, left, right, bottom, top):
        super(HeatWindow, self).__init__(caption = '', width = SCREEN_W, height = SCREEN_H)

        if bottom > top:
            (bottom, top) = (top, bottom)
            self.invert_y = True
        else:
            self.invert_y = False

        if left > right:
            (left, right) = (right, left)
            self.invert_x = True
        else:
            self.invert_x = False

        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

        # Check if we need to reinitialize the OpenGL state
        if (self.width != abs(self.right - self.left) or
            self.height != abs(self.top - self.bottom) or
            None in (self.fbo, self.texture, self.palette)):
            self.prepare(abs(right - left), abs(top - bottom))

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(left, right, bottom, top)

        self._clear_framebuffer()

        points = [(random.gauss(.5, .08) * 100, random.gauss(.5, .075) * 100) for i in xrange(100)]
        self.add_points(points, 25)


    def on_draw(self):
        pass


    # compileProgram in OpenGL.GL.shaders fails to validate if multiple samplers are used
    def compileProgram(*shaders):
        program = glCreateProgram()
        for shader in shaders:
            glAttachShader(program, shader)

        glLinkProgram(program)

        for shader in shaders:
            glDeleteShader(shader)

        return program

    @classmethod
    def cleanup(cls):
        if cls.fbo: glDeleteFramebuffersEXT(cls.fbo)
        if cls.texture: glDeleteTextures(cls.texture)
        if cls.palette: glDeleteTextures(cls.palette)
        cls.fbo = cls.texture = cls.palette = None

    @classmethod
    def prepare(cls, width, height):
        cls.cleanup()

        cls.width = width
        cls.height = height

        # Render Flags
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_1D)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnableClientState(GL_VERTEX_ARRAY)

        cls._compile_programs()
        cls._load_palette()
        cls._create_framebuffer()

    @classmethod
    def _load_palette(cls, path='palettes/classic.png'):
        image = PIL.Image.open(path)
        cls.palette = gl.GLuint()
        glGenTextures(1, byref(cls.palette))
        glActiveTextureARB(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, cls.palette)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB,
                     image.size[1],
                     0, GL_RGB, GL_UNSIGNED_BYTE,
                     image.tostring('raw', 'RGB', 0, -1))

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP)



    @classmethod
    def _create_framebuffer(cls):
        cls.texture = gl.GLuint()
        glGenTextures(1, byref(cls.texture))
        cls.fbo = gl.GLuint()
        glGenFramebuffersEXT(1, byref(cls.fbo))

        glActiveTextureARB(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, cls.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     cls.width, cls.height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, cls.fbo)
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                                  GL_TEXTURE_2D, cls.texture, 0)

        status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)
        assert status == GL_FRAMEBUFFER_COMPLETE_EXT, status

    @classmethod
    def _clear_framebuffer(cls):
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, cls.fbo)
        glClear(GL_COLOR_BUFFER_BIT)

    @classmethod
    def _compile_programs(cls):
        # Shader program to transform color into the proper palette based on the alpha channel

        cls.color_transform_program = compileProgram(
            compileShader('''
                void main() {
                    gl_Position = ftransform();
                }
            ''', GL_VERTEX_SHADER),
            compileShader('''
                uniform float alpha;
                uniform sampler1D palette;
                uniform sampler2D framebuffer;
                uniform vec2 windowSize;

                void main() {
                    //gl_FragColor.rgb = texture1D(palette, texture2D(framebuffer,  gl_FragCoord.xy / windowSize).a).rgb;
                    gl_FragColor.a = alpha;
                }
            ''', GL_FRAGMENT_SHADER))

        # Shader program to place heat points
        cls.faded_points_program = compileProgram(
            compileShader('''
                uniform float r;
                attribute vec2 point;
                varying vec2 center;

                void main() {
                    gl_Position = ftransform();
                    center = point;
                }
            ''', GL_VERTEX_SHADER),
            compileShader('''
                uniform float r;
                varying vec2 center;

                void main() {
                    float d = distance(gl_FragCoord.xy, center);
                    if (d > r) discard;

                    gl_FragColor.rgb = vec3(1.0, 1.0, 1.0);
                    gl_FragColor.a = (0.5 + cos(d * 3.14159265 / r) * 0.5) * 0.25;

                    // Alternate fading algorithms
                    //gl_FragColor.a = (1.0 - (log(1.1+d) / log(1.1+r)));
                    //gl_FragColor.a = (1.0 - (pow(d, 0.5) / pow(r, 0.5)));
                    //gl_FragColor.a = (1.0 - ((d*d) / (r*r))) / 2.0;
                    //gl_FragColor.a = (1.0 - (d / r)) / 2.0;

                    // Clamp the alpha to the range [0.0, 1.0]
                    gl_FragColor.a = clamp(gl_FragColor.a, 0.0, 1.0);
                }
            ''', GL_FRAGMENT_SHADER))

    def add_points(self, points, radius):
        # Render all points with the specified radius
        glUseProgram(self.faded_points_program)
        glUniform1f(glGetUniformLocation(self.faded_points_program, 'r'), radius)

        point_attrib_location = glGetAttribLocation(self.faded_points_program, 'point')
        glEnableVertexAttribArray(point_attrib_location)
        glVertexAttribPointer(point_attrib_location, 2, GL_FLOAT, False, 0,
                              struct.pack("ff" * 4 * len(points),
                                          *(val for (x, y) in points
                                                for val in (x - self.left, y - self.bottom) * 4)))

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        vertices = [point for (x, y) in points
                          for point in ((x + radius, y + radius), (x - radius, y + radius),
                                        (x - radius, y - radius), (x + radius, y - radius))]
        glVertexPointer(vertices)
        glDrawArrays(GL_QUADS, 0, len(vertices))
        glFlush()

        glDisableVertexAttribArray(point_attrib_location)

    def transform_color(self, alpha):
        # Transform the color into the proper palette
        glUseProgram(self.color_transform_program)
        glUniform1f(glGetUniformLocation(self.color_transform_program, 'alpha'), alpha)
        glUniform1i(glGetUniformLocation(self.color_transform_program, 'palette'), 0)
        glUniform1i(glGetUniformLocation(self.color_transform_program, 'framebuffer'), 1)
        glUniform2f(glGetUniformLocation(self.color_transform_program, 'windowSize'), self.width, self.height)

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        glBlendFunc(GL_ONE, GL_ZERO)

        vertices = [(self.left, self.bottom), (self.right, self.bottom),
                    (self.right, self.top), (self.left, self.top)]
        glVertexPointer(vertices)
        glDrawArrays(GL_QUADS, 0, len(vertices))
        glFlush()

    def get_image(self):
        # Get the data from the heatmap framebuffer and convert it into a PIL image
        glActiveTextureARB(GL_TEXTURE1)
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
        im = PIL.Image.frombuffer('RGBA', (self.width, self.height), data, 'raw', 'RGBA', 0, (1 if self.invert_y else -1))

        if self.invert_x:
            im.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Write the image to a buffer as a PNG
        f = cStringIO.StringIO()
        im.save(f, 'png')
        f.seek(0)

        return f

 
window = HeatWindow(0, SCREEN_W, 0, SCREEN_H)
pyglet.app.run()
