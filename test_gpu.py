from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_l',
    root=r'D:\AI\models\insightface',
    providers=['CUDAExecutionProvider']
)

app.prepare(ctx_id=0)

print("GPU OK")