## Peg Solitaire RL

Run by:
`python main.py`

Command-line options
```
-g,     --graphics      FLAG: activates drawing of the board during training.

    usage:
    > python main.py -g

-i,     --interactive   FLAG: puts the environment in a state suceptible for input from the user

    usage:
    > python main.py -i

-cfg    --config        ARG:  load a specified config file.
       
    usage:
    > python main.py -cfg=some_config.yml
```

#### Interactive mode
Let the user interact with the peg board by numpad (while NOT in num-lock mode)

        Selection controls:

        |  7 - up left   |                |  9 - up right   |
        |  4 - left      | 5 - toggle peg |  6 - right      |
        |  1 - down left |                |  3 - down right |
        
        p:      print current board possible number of actions and score
        enter:  start training

if selection is invisible, set outline_color in config file to something different to [255,255,255]

#### Config file

The config file controls all the hyper-parameters of our RL models.
Additionally it controls some of the visual aspects of our application
