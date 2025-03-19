from models_config import load_clip

#i already have a search class in vectordb.py

class SearchBase(ABC):
    """Base class for all search algorithms."""
    def __init__(self, vectordb):
        self.vectordb = vectordb

    @abstractmethod
    def search(self, query, k: int = 10) -> list[str]:
        """
        Search for similar images.
        
        Args:
            query: Query (can be image path, text, or other)
            k: Number of results to return
            
        Returns:
            List of (image_path, score) tuples
        """
        pass 

class ClipSearch(SearchBase):
    """Search using CLIP model."""
    def __init__(self, vectordb, model_name="clip"):
        super().__init__(vectordb)
        self.model_name = model_name

    def search(self, query_text, k=5):\
        """
        Search images using text descriptions via CLIP.

        Args:
            query_text: Text description of the query
            k: Number of results to return

        Returns:
            List of (image_path, score) tuples
        """
        
        
