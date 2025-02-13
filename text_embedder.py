from typing import List, Optional, Union
import numpy as np
from openai import OpenAI

class TextEmbedder:
    """A class to handle text embedding operations using OpenAI's API"""

    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initialize the TextEmbedder.

        Args:
            model (str): The OpenAI embedding model to use.
                        Defaults to "text-embedding-3-large".
        """
        self.client = OpenAI()
        self.model = model

    def get_reference_dictionary(self, texts: List[str]):
        reference_dict = {}
        for text in texts:
            reference_dict[text] = self.get_embedding(text)
        return reference_dict        

    def get_embedding(self,
                     text: Union[str, List[str]],
                     normalize: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get embeddings for one or more texts using OpenAI's API.

        Args:
            text (Union[str, List[str]]): Single text string or list of text strings
                                        to get embeddings for.
            normalize (bool): Whether to normalize the resulting vectors.
                            Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The embedding vector(s) as numpy array(s).
            For single text input, returns a single numpy array.
            For list input, returns a list of numpy arrays.

        Raises:
            Exception: If there's an error calling the OpenAI API.
        """
        # Convert single string to list for consistent processing
        input_texts = [text] if isinstance(text, str) else text

        try:
            print(f"Calling OpenAI API for {len(input_texts)} text embedding(s).")
            response = self.client.embeddings.create(
                model=self.model,
                input=input_texts
            )

            # Convert embeddings to numpy arrays
            embeddings = [np.array(data.embedding, dtype=np.float32)
                        for data in response.data]

            # Normalize if requested
            if normalize:
                embeddings = [self._normalize(emb) for emb in embeddings]

            # Return single array for single input, list for multiple inputs
            return embeddings[0] if isinstance(text, str) else embeddings

        except Exception as e:
            raise Exception(f"Error creating embedding: {e}")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            vector (np.ndarray): The vector to normalize.

        Returns:
            np.ndarray: The normalized vector.
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
