from typing import Optional, OrderedDict, Callable

import numpy as np

from meal.visualization.types import TileKey


class TileCache:
    def __init__(self, max_items: int = 2048):
        self._max = max_items
        self._store: "OrderedDict[TileKey, np.ndarray]" = OrderedDict()

    def get(self, key: TileKey) -> Optional[np.ndarray]:
        img = self._store.get(key)
        if img is not None:
            # LRU touch
            self._store.move_to_end(key)
        return img

    def put(self, key: TileKey, img: np.ndarray) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = img
        if len(self._store) > self._max:
            self._store.popitem(last=False)  # evict LRU

    def get_or_render(self, key: TileKey, render_fn: Callable[[], np.ndarray]) -> np.ndarray:
        img = self.get(key)
        if img is not None:
            return img
        img = render_fn()
        self.put(key, img)
        return img
