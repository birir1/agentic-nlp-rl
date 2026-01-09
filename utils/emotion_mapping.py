"""
Emotion → Valence mapping utilities
"""

def emotion_to_valence(emotion_input):
    """
    Convert emotion output to a scalar valence.

    Supports:
    1) string emotion labels
    2) probability dictionaries {emotion: prob}
    """

    # Case 1: probability distribution
    if isinstance(emotion_input, dict):
        mapping = {
            "joy": 1.0,
            "happiness": 0.8,
            "neutral": 0.0,
            "sadness": -0.6,
            "anger": -0.8,
            "fear": -0.7,
            "disgust": -0.9,
            "surprise": 0.2,
        }

        valence = 0.0
        for emotion, prob in emotion_input.items():
            valence += mapping.get(emotion.lower(), 0.0) * prob
        return valence

    # Case 2: string label
    if isinstance(emotion_input, str):
        emotion = emotion_input.lower().strip()
        return {
            "joy": 1.0,
            "happiness": 0.8,
            "neutral": 0.0,
            "sadness": -0.6,
            "anger": -0.8,
            "fear": -0.7,
            "disgust": -0.9,
            "surprise": 0.2,
        }.get(emotion, 0.0)

    # Fallback
    return 0.0
