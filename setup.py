import setuptools

setuptools.setup(
    name="torchbeast",
    packages=["torchbeast"],
    version="0.0.13",
    author="TorchBeast team",
    install_requires=[],
    entry_points={
        "BaseExploit": [
            "multitasktruncateexploit = torchbeast.pbt_exploit:MultitaskTruncateExploit",
            "multitaskbacktrackexploit = torchbeast.pbt_exploit:MultitaskBacktrackExploit",
        ]
    },
)
