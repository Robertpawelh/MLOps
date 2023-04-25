from pydantic import BaseModel

class PredictionInput(BaseModel):
    image_url: str

    class Config:
        schema_extra = {
            'example': {
                'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dogge_Odin.jpg/800px-Dogge_Odin.jpg'
            }
        }

class PredictionOutput(BaseModel):
    cat: float
    dog: float

    class Config:
        schema_extra = {
            'example': {
                'cat': 0.999,
                'dog': 0.001
            }
        }
