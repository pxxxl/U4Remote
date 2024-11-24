import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_anc
import time
import tkinter as tk
from tkinter import filedialog
import os
import sys
import json
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from arguments import ModelParams


# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


g_camera = util.Camera(1080, 1920)
BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7
g_FVV_path=""
VIDEO_FPS = 25.0
VIDEO_INTERVAL = 1.0 / VIDEO_FPS

g_last_frame_time = 0.0
g_timestep = 0
g_paused = True
g_reset = False
g_total_frame = 250
def impl_glfw_init():
    window_name = "Tiny i3DV Viewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def init_activated_renderer_state(gaus: util_anc.GaussianModel):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def update_activated_renderer_state(gaus: util_anc.GaussianModel):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def main(args_param, dataset, base_path, scene_path):
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables, \
        g_FVV_path, g_paused, g_reset, g_timestep, g_last_frame_time, g_total_frame
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    # try:
    from renderer_cuda import CUDARenderer
    g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    # except ImportError:
    #     pass
    
    g_renderer_idx = BACKEND_CUDA
    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_anc.GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
    )

    frame_path = "frame000000"
    model_path = os.path.join(base_path, scene_path, frame_path)
    loaded_iter = 30000
    gaussians.load_ply_sparse_gaussian(os.path.join(model_path,
                                "point_cloud",
                                "iteration_" + str(loaded_iter),
                                "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(os.path.join(model_path,
                                "point_cloud",
                                "iteration_" + str(loaded_iter),
                                "checkpoint.pth"))
    gaussians.initial_for_P_frame(args_param.ntc_cfg)

    init_activated_renderer_state(gaussians)

    g_last_frame_time=time.time()
    
    # settings
    frm_idx = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        current_time=time.time()
        # if current_time - g_last_frame_time >= VIDEO_INTERVAL and not g_paused and g_timestep < g_total_frame-1:
        #     g_timestep+=1
        #     g_last_frame_time = current_time
        # if g_reset:
            # g_renderer.fvv_reset()
            # g_reset = False
            # g_last_frame_time = time.time()
        g_renderer.draw(g_camera, g_timestep)
        print(frm_idx, time.time())

        loaded_iter = 30000 if frm_idx == 0 else 1500
        model_path = os.path.join(base_path, scene_path, f"frame{frm_idx:06d}")
        if not g_paused:
            if frm_idx == 0:
                gaussians.load_ply_sparse_gaussian(os.path.join(model_path,
                                            "point_cloud",
                                            "iteration_" + str(loaded_iter),
                                            "point_cloud.ply"))
                gaussians.load_mlp_checkpoints(os.path.join(model_path,
                                            "point_cloud",
                                            "iteration_" + str(loaded_iter),
                                            "checkpoint.pth"))
            else:
                gaussians.load_ntc_checkpoints(os.path.join(model_path+"_offsets",
                                            "point_cloud",
                                            "iteration_" + str(loaded_iter*2),
                                            "NTC.pth"), stage="stage1")
                gaussians.load_ntc_checkpoints(os.path.join(model_path,
                                            "point_cloud",
                                            "iteration_" + str(loaded_iter),
                                            "NTC.pth"), stage="stage2")
                gaussians.update_by_ntc()                            
        update_activated_renderer_state(gaussians)

        if g_paused != True:
            
            frm_idx = frm_idx + 1 if frm_idx < g_total_frame -1 else 0

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)

                imgui.text(f"# of Anchors = {len(gaussians._anchor)}")

                imgui.text(f"Render FPS = {imgui.get_io().framerate:.1f}")
                imgui.text(f"Video FPS = {VIDEO_FPS:.1f}")
                imgui.text(f"FVV Dir:{g_FVV_path}")
                imgui.text(f"Frame {g_timestep}")
                imgui.text(f"#Frames: ")
                imgui.same_line()
                total_frame_changed, g_total_frame = imgui.slider_int(
                    "frames", g_total_frame, 1, 250
                )
                if imgui.button("Pause"):
                    g_paused = True 
                    g_last_frame_time=time.time()
                
                imgui.same_line()
                
                if imgui.button("Play"):
                    g_paused = False
                    g_last_frame_time=time.time()
                    
                imgui.same_line() 
                
                if imgui.button("Reset"):
                    g_paused = True
                    g_reset = True
                    g_timestep=0
                    g_last_frame_time=time.time() 

                imgui.same_line() 
                
                if imgui.button("Step"):
                    frm_idx+=1
                    g_last_frame_time=time.time()                     
                
                if imgui.button(label='load ply'):
                    file_path = filedialog.askopenfilename(title="load ply",
                        initialdir="/amax/yangjy/workspace/crosscheck/outputs_AVS/",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            # gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                        except RuntimeError as e:
                            pass
                
                imgui.same_line()

                # if imgui.button(label='save ply'):
                #     file_path = filedialog.asksaveasfilename(title="save ply",
                #         initialdir="/amax/yangjy/workspace/crosscheck/outputs_AVS/",
                #         defaultextension=".txt",
                #         filetypes=[('ply file', '.ply')]
                #         )
                #     if file_path:
                #         try:
                #             util_3dgstream.save_gau_cuda(g_renderer.gaussians, file_path)
                #         except RuntimeError as e:
                #             pass
                
                # imgui.same_line()
                                        
                # if imgui.button(label='load FVV'):
                #     dir_path = filedialog.askdirectory(title="load FVV",
                #         initialdir="/amax/yangjy/workspace/crosscheck/outputs_AVS/"
                #         )
                #     if dir_path:
                #         try:
                #             g_FVV_path = dir_path
                #             g_renderer.NTCs = util_3dgstream.load_NTCs(g_FVV_path, g_renderer.gaussians, g_total_frame)
                #             g_renderer.additional_3dgs = util_3dgstream.load_Additions(g_FVV_path, g_total_frame)
                #         except RuntimeError as e:
                #             pass                
                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)
                
                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                        "auto sort", g_auto_sort,
                    )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3
                    stride = nrChannels * width
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )
                    # Add buttons directly in the main menu bar for control actions
                    
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny i3DV Viewer.")
    lp = ModelParams(parser)
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--config_path", type=str, default = "config.json")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args(sys.argv[1:])
    
    assert args.config_path is not None, "Please provide a config path"
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    for key, value in config["model_params"].items():
        setattr(args, key, value)
    args.ntc_cfg = config["ntc_cfg"]

    base_path = "D:\\AVS_sequences\\visual"
    scene_path = "0.0001_12.0_dh"
    # scene_path = "0.001_1.0_sg"
    # scene_path = "0.004_2.0_gz"
    # scene_path = "0.004_1.0_dg4"


    main(args, lp.extract(args), base_path, scene_path)
