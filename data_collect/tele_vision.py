import os
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
import time
from multiprocessing import Process, Array, Value, shared_memory
import numpy as np
import asyncio

class OpenTeleVision:
    def __init__(self, img_shape, shm_name, stereo=True, cert_file=os.path.join(os.path.dirname(__file__), "cert.pem"), key_file=os.path.join(os.path.dirname(__file__), "key.pem")):
        self.stereo = stereo
        if self.stereo:
            self.img_shape = (img_shape[0], 2*img_shape[1], 3)
        else:
            self.img_shape = (img_shape[0], img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]

        self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False))

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        self.app.spawn(start=False)(self.main)
        self.shm_name = shm_name
    
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
        self.app.spawn(start=False)(self.main)

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    def run(self):
        self.app.run()

    async def on_cam_move(self, event, session):
        # Update head matrix data
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
        except Exception as e:
            # print(f"Error updating head_matrix_shared: {e}")
            pass
        
        # Update aspect data
        try:
            self.aspect_shared.value = event.value['camera']['aspect']
        except Exception as e:
            # print(f"Error updating aspect_shared: {e}")
            pass

    async def on_hand_move(self, event, session):
        # Update left hand data
        try:
            self.left_hand_shared[:] = event.value["leftHand"]
        except Exception as e:
            # print(f"Error updating left_hand_shared: {e}")
            pass
        
        # Update right hand data
        try:
            self.right_hand_shared[:] = event.value["rightHand"]
        except Exception as e:
            # print(f"Error updating right_hand_shared: {e}")
            pass
        
        # Update left landmarks data
        try:
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
        except Exception as e:
            # print(f"Error updating left_landmarks_shared: {e}")
            pass
        
        # Update right landmarks data
        try:
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except Exception as e:
            # print(f"Error updating right_landmarks_shared: {e}")
            pass

    async def main(self, session, fps=60):
        try:
            session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=True, showRight=True)
        except Exception as e:
            print(f"Failed to initialize hands: {e}")
            return
            
        end_time = 0
        while True:
            try:
                start_time = time.time()
                display_image = self.img_array
                if not self.stereo:
                    session.upsert(
                    ImageBackground(
                        display_image[:, :self.img_width],
                        format="jpeg",
                        quality=60,
                        key="left-image",
                        interpolate=True,
                        aspect=1.778,
                        distanceToCamera=2,
                        position=[0, -0.5, -2],
                        rotation=[0, 0, 0],
                    ),
                    to="bgChildren",
                    )
                    # print('fps', 1 / (time.time() - end_time))
                    # end_time = time.time()
                    rest_time = 1/fps - time.time() + start_time
                else:
                    session.upsert(
                    [ImageBackground(
                        # Can scale the images down.
                        display_image[::2, :self.img_width],
                        # display_image[:self.img_height:2, ::2],
                        # 'jpg' encoding is significantly faster than 'png'.
                        format="jpeg",
                        quality=75,
                        key="left-image",
                        interpolate=True,
                        # fixed=True,
                        aspect=1.778,
                        # distanceToCamera=0.5,
                        height = 8,
                        position=[0, -1, 3],
                        # rotation=[0, 0, 0],
                        layers=1, 
                        alphaSrc="./vinette.jpg"
                    ),
                    ImageBackground(
                        # Can scale the images down.
                        display_image[::2, self.img_width:],
                        # display_image[self.img_height::2, ::2],
                        # 'jpg' encoding is significantly faster than 'png'.
                        format="jpeg",
                        quality=75,
                        key="right-image",
                        interpolate=True,
                        # fixed=True,
                        aspect=1.778,
                        # distanceToCamera=0.5,
                        height = 8,
                        position=[0, -1, 3],
                        # rotation=[0, 0, 0],
                        layers=2, 
                        alphaSrc="./vinette.jpg"
                    )],
                    to="bgChildren",
                    )
                    # print('fps', 1 / (time.time() - end_time))
                    end_time = time.time()
                    rest_time = 1/fps - time.time() + start_time
                # rest_time = 0.03 -> 30hz
                await asyncio.sleep(rest_time)
            except Exception as e:
                print(f"WebSocket error in main loop: {e}")
                # If session is missing, break out of the loop to wait for reconnection
                if "Websocket session is missing" in str(e):
                    print("WebSocket session disconnected, exiting main loop")
                    break
                # For other errors, wait a bit before retrying
                await asyncio.sleep(0.1)
                continue

    # def modify_shared_image(self, img, random=False):
    #     assert img.shape == self.img_shape, f"Image shape must be {self.img_shape}, got {img.shape}"
    #     existing_shm = shared_memory.SharedMemory(name=self.shm_name)
    #     shared_image = np.ndarray(self.img_shape, dtype=np.uint8, buffer=existing_shm.buf)
    #     shared_image[:] = img[:] if not random else np.random.randint(0, 256, self.img_shape, dtype=np.uint8)
    #     existing_shm.close()

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)
