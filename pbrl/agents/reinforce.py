#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DummyAgent:
    
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        pass

    def __call__(self) -> str:
        return 'Hello World'
