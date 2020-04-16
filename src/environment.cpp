/*  Copyright © 2018, Roboti LLC

    This file is licensed under the MuJoCo Resource License (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.roboti.us/resourcelicense.txt
*/

#include "../include/mujoco.h"
#include "../include/mjxmacro.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <tuple>
#include <iostream>
#include <math.h>

// select EGL, OSMESA or GLFW
#if defined(MJ_EGL)
    #include <EGL/egl.h>
#elif defined(MJ_OSMESA)
    #include <GL/osmesa.h>
    OSMesaContext ctx;
    unsigned char buffer[10000000];
#else
    #include "glfw3.h"
#endif

mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

class Environment
{  
    public:

        FILE* fp;
        int W,H;
        int framecount = 0;
        mjrRect viewport;
        unsigned char* rgb;
        float* depth;

        Environment(const char* model_name, bool record = false)
        {
            std::cout << model_name << std::endl;
            // activate software
            mj_activate("mjkey.txt");

            // load and compile model
            char error[1000] = "Could not load binary model";
            m = mj_loadXML(model_name, 0, error, 1000);
            
            std::cout << "model loaded" << std::endl;

            if( !m )
                mju_error_s("Load model error: %s", error);

            // make data
            d = mj_makeData(m);

            std::cout << "num deg free: " << m->nv<< std::endl;
            std::cout << "num gen coor: " << m->nq << std::endl;
            std::cout << "num act: " << m->nu << std::endl;

            std::cout << "q0: " << d->qpos[0] << std::endl;
            std::cout << "v0: " << d->qvel[0] << std::endl;

            // initialize visualization data structures
            mjv_defaultCamera(&cam);
            mjv_defaultOption(&opt);
            mjv_defaultScene(&scn);
            mjr_defaultContext(&con);

            // center and scale view
            cam.lookat[0] = m->stat.center[0];
            cam.lookat[1] = m->stat.center[1];
            cam.lookat[2] = m->stat.center[2];
            cam.distance = 1.5 * m->stat.extent;

            // create scene and context
            mjv_makeScene(m, &scn, 2000);
            
            std::cout << "env constructed" << std::endl;

            if (record)
            {
                std::cout << "this environment is recording" << std::endl;
                // initialize OpenGL
                initOpenGL();

                std::cout << "TEST"<< std::endl;
                mjr_makeContext(m, &con, mjFONTSCALE_150);

                // set rendering to offscreen buffer
                mjr_setBuffer(mjFB_OFFSCREEN, &con);
                if( con.currentBuffer!=mjFB_OFFSCREEN )
                    printf("Warning: offscreen rendering not supported, using default/window framebuffer\n");

                // get size of active renderbuffer
                viewport =  mjr_maxViewport(&con);
                W = viewport.width;
                H = viewport.height;

                // allocate rgb and depth buffers
                rgb = (unsigned char*)malloc(3*W*H);
                depth = (float*)malloc(sizeof(float)*W*H);
                if( !rgb || !depth )
                    mju_error("Could not allocate buffers");

                // create output rgb file
                fp = fopen("outputRGB", "wb");
                if( !fp )
                    mju_error("Could not open rgbfile for writing");

                monitor(m,d);
            }
                        
        }

        double remap (double value, double from1, double to1, double from2, double to2) {
            return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
        }

        void set_randomStart(double start_pos, double start_vel)
        {
            d->qpos[0] = start_pos;
            d->qvel[0] = start_vel;
        }

        /*
         * Step function takes in input an action, applies the action in the environment
         * and returns a reward and the next state
         * */
        std::tuple<mjtNum, std::tuple<double, mjtNum>>  step(bool record = false, double action = 0.0)
        {
            assert(action <= 2.0 || action >= -2.0);

            if(record)
                monitor(m,d);

            mj_step(m, d);

            // apply our controls here instead of using the mjcb_control callback
            d->ctrl[0] = action;

            // taking extra steps in the environment
            for (int i = 0; i < 5; i++)
            {
                mj_step(m, d);
                if(record)
                    monitor(m,d);
            }

            d->qpos[0] = std::fmod(d->qpos[0], 2*M_PI);

            double theta = 0.0;
            if (d->qpos[0] < 0.0)
                theta = remap(d->qpos[0], 0.0, -2*M_PI, -M_PI, M_PI);
            else
                theta = remap(d->qpos[0], 0.0, 2*M_PI, -M_PI, M_PI);

            std::tuple<double, mjtNum> next_state = std::make_tuple(d->qpos[0] , d->qvel[0]);

            mjtNum reward = - std::pow( std::fmod(theta + M_PI, 2*M_PI) - M_PI, 2)
                            - 0.05*std::pow(d->qvel[0], 2)
                            - 0.3*std::pow(action, 2);

            std::tuple<mjtNum, std::tuple<mjtNum, mjtNum>>  ret = std::make_tuple(reward, next_state);
            return ret;
        }

        void reset()
        {
            std::cout << "env reset" << std::endl;
            mj_resetData(m, d);
            mj_forward(m, d);
        }

        void stop()
        {
            std::cout << "env stopped" << std::endl;
            //free visualization storage
            mjv_freeScene(&scn);
            mjr_freeContext(&con);

            // free MuJoCo model and data, deactivate
            mj_deleteData(d);
            mj_deleteModel(m);
            mj_deactivate();

            // close file, free buffers
            fclose(fp);
            free(rgb);
            free(depth);

            // close OpenGL
            closeOpenGL();
        }
        
        // create OpenGL context/window
        void initOpenGL(void)
        {
            //------------------------ EGL
        #if defined(MJ_EGL)
            // desired config
            const EGLint configAttribs[] ={
                EGL_RED_SIZE,           8,
                EGL_GREEN_SIZE,         8,
                EGL_BLUE_SIZE,          8,
                EGL_ALPHA_SIZE,         8,
                EGL_DEPTH_SIZE,         24,
                EGL_STENCIL_SIZE,       8,
                EGL_COLOR_BUFFER_TYPE,  EGL_RGB_BUFFER,
                EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE,    EGL_OPENGL_BIT,
                EGL_NONE
            };

            // get default display
            EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if( eglDpy==EGL_NO_DISPLAY )
                mju_error_i("Could not get EGL display, error 0x%x\n", eglGetError());

            // initialize
            EGLint major, minor;
            if( eglInitialize(eglDpy, &major, &minor)!=EGL_TRUE )
                mju_error_i("Could not initialize EGL, error 0x%x\n", eglGetError());

            // choose config
            EGLint numConfigs;
            EGLConfig eglCfg;
            if( eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs)!=EGL_TRUE )
                mju_error_i("Could not choose EGL config, error 0x%x\n", eglGetError());

            // bind OpenGL API
            if( eglBindAPI(EGL_OPENGL_API)!=EGL_TRUE )
                mju_error_i("Could not bind EGL OpenGL API, error 0x%x\n", eglGetError());

            // create context
            EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
            if( eglCtx==EGL_NO_CONTEXT )
                mju_error_i("Could not create EGL context, error 0x%x\n", eglGetError());

            // make context current, no surface (let OpenGL handle FBO)
            if( eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx)!=EGL_TRUE )
                mju_error_i("Could not make EGL context current, error 0x%x\n", eglGetError());

            //------------------------ OSMESA
        #elif defined(MJ_OSMESA)
            // create context
            ctx = OSMesaCreateContextExt(GL_RGBA, 24, 8, 8, 0);
            if( !ctx )
                mju_error("OSMesa context creation failed");

            // make current
            if( !OSMesaMakeCurrent(ctx, buffer, GL_UNSIGNED_BYTE, 800, 800) )
                mju_error("OSMesa make current failed");

            //------------------------ GLFW
        #else
            // init GLFW
            if( !glfwInit() )
                mju_error("Could not initialize GLFW");

            // create invisible window, single-buffered
            glfwWindowHint(GLFW_VISIBLE, 0);
            glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
            GLFWwindow* window = glfwCreateWindow(800, 800, "Invisible window", NULL, NULL);
            if( !window )
                mju_error("Could not create GLFW window");

            // make context current
            glfwMakeContextCurrent(window);
        #endif
        }

        // close OpenGL context/window
        void closeOpenGL(void)
        {
            //------------------------ EGL
        #if defined(MJ_EGL)
            // get current display
            EGLDisplay eglDpy = eglGetCurrentDisplay();
            if( eglDpy==EGL_NO_DISPLAY )
                return;

            // get current context
            EGLContext eglCtx = eglGetCurrentContext();

            // release context
            eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

            // destroy context if valid
            if( eglCtx!=EGL_NO_CONTEXT )
                eglDestroyContext(eglDpy, eglCtx);

            // terminate display
            eglTerminate(eglDpy);

            //------------------------ OSMESA
        #elif defined(MJ_OSMESA)
            OSMesaDestroyContext(ctx);

            //------------------------ GLFW
        #else
            // terminate GLFW (crashes with Linux NVidia drivers)
            #if defined(__APPLE__) || defined(_WIN32)
                glfwTerminate();
            #endif
        #endif
        }

        void monitor(const mjModel* m, mjData* d)
        {
            // update abstract scene
            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

            // render scene in offscreen buffer
            mjr_render(viewport, &scn, &con);

            // add time stamp in upper-left corner
            char stamp[50];
            sprintf(stamp, "Time = %.3f", d->time);
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, stamp, NULL, &con);

            // read rgb and depth buffers
            mjr_readPixels(rgb, depth, viewport, &con);

            // insert subsampled depth image in lower-left corner of rgb image
            const int NS = 3;           // depth image sub-sampling
            for( int r=0; r<H; r+=NS )
                for( int c=0; c<W; c+=NS )
                {
                    int adr = (r/NS)*W + c/NS;
                    rgb[3*adr] = rgb[3*adr+1] = rgb[3*adr+2] =
                        (unsigned char)((1.0f-depth[r*W+c])*255.0f);
                }

            // write rgb image to file
            fwrite(rgb, 3, W*H, fp);

            // print every 10 frames: '.' if ok, 'x' if OpenGL error
            if( ((framecount++)%10)==0 )
            {
                if( mjr_getError() )
                    printf("x");
                else
                    printf(".");
            }
        }

    private:
    // MuJoCo data structures
    mjModel* m = NULL;                  // MuJoCo model
    mjData* d = NULL;                   // MuJoCo data

};