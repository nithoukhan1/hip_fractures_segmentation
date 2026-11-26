# results/_inbox/

Drop the `results_<RUN>.zip` files downloaded from Kaggle here. Download them from the finished version: *Output* tab → `results_<RUN>.zip`.

Then either tell Claude in the project thread, or run this from the project folder:
```
python scripts/register_run.py results/_inbox/results_<RUN>.zip --version <Kaggle version number> --purpose "<one line why>"
```
The script copies the run into `results/<RUN>/`, adds a `NOTES.md` stub, appends a line to `results/EXPERIMENTS.md` and moves the zip to `results/_inbox/filed/`.

Nothing in this folder is committed to git except this README (see `.gitignore`); only the filed run folders in `results/<RUN>/` are.
