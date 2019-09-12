import unittest
import face_alignment
from scipy import io as mat

class Tester(unittest.TestCase):
    def test_predict_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True)
        preds = fa.get_landmarks('test/assets/Surprise_363_1.jpg',all_faces=True)          
        savename = 'test/assets/Surprise_363_1' + '_lms.mat'
        if preds is not None:        
            mat.savemat(savename, mdict={'finelms': preds})

if __name__ == '__main__':
    unittest.main()
