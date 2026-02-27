# Code Validation

Hi Kilo Code agent! Whatever model you are using, make sure you validate the code you generated.

## Steps to Validate

Run these commands in sequence:
1. `pixi run fmt`: format your code. This ensures that the code written by human and by you (coding agent) are consistent, following the style that Rust has provided.
2. `pixi run lint`: ensure no linter errors.
3. `pixi run lint-fix`: if there are any linter error, fix it with this command.
4. `pixi run lint`: recheck again, maybe there are linter errors that need manual fix.
5. `pixi run test`: build the code, then run all unit tests. This ensures that the code you generate pass all the unit tests.
