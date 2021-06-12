# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [2.0.0](https://github.com/rphln/Mawile/compare/v1.0.0...v2.0.0) (2021-06-12)


### ⚠ BREAKING CHANGES

* **MemoryPlayer:** Refactor the memory handler.

### Other changes

* Add `.versionrc`. ([2f2f376](https://github.com/rphln/Mawile/commit/2f2f376df14fe05b81939363a254d5343afeaee3))
* Update `.gitignore`. ([c5534d1](https://github.com/rphln/Mawile/commit/c5534d182f41675b1d0cf3b0b28b98d7687dccd4))


### Minor changes

* **Common:** Move `reward_computing_helper` to `common.py`. ([c46babc](https://github.com/rphln/Mawile/commit/c46babc4a40bdc63640e3ebb7a168fe0d568d276))
* **MemoryPlayer:** Add more typing information. ([d9048bc](https://github.com/rphln/Mawile/commit/d9048bcd6deb8c3d4f4b325152bc52b2b1d31c99))
* **NaivePlayer:** Move to `players.py`. ([eb2addc](https://github.com/rphln/Mawile/commit/eb2addc6b46e15120820d9e8c994ef60f98ae631))
* **Train:** Save the model as `checkpoint.keras`. ([d82fd9b](https://github.com/rphln/Mawile/commit/d82fd9bc5978a5b2d4b5dbeaea0fc6301c70d19a))
* **Train:** Use new players on pre-training. ([525c126](https://github.com/rphln/Mawile/commit/525c1260d1bc83b31dee7bdcd5e78e0fbb736826))


### Major changes

* **MemoryPlayer:** Refactor the memory handler. ([be8ce84](https://github.com/rphln/Mawile/commit/be8ce84eac890d1c36914ad1f2d0ec57e823b87b))
* **MemoryPlayer:** Use a named tuple for transitions. ([5fd3a3f](https://github.com/rphln/Mawile/commit/5fd3a3f79e150400da055a960384caeca948c757))

## 1.0.0 (2021-06-12)


### Features

* **DenseQPlayer:** Update `reward_computing_helper`. ([6a047f7](https://github.com/rphln/Mawile/commit/6a047f779ba79fc102e274e07a22602d02897a92))
* **MemoryPlayer:** Add `max_concurrent_battles`. ([bcfafb1](https://github.com/rphln/Mawile/commit/bcfafb1926b0dd1fc8909c06059bcdd3f5636204))
* **Train:** Add a pre-training phase. ([721ef26](https://github.com/rphln/Mawile/commit/721ef26e17900ea213cec3568658e4d5d28a3fb5))
* **Train:** Change the match statistics tracker. ([0d9cf2f](https://github.com/rphln/Mawile/commit/0d9cf2fdd88e68cf8027b8b801974d0e560c9432))
* Initial commit. ([e935c29](https://github.com/rphln/Mawile/commit/e935c29f8bc52450ae3c13bc095f3c4c557f5da8))


### Bug Fixes

* **DenseQPlayer:** Change the default `exploration_rate`. ([dc21d92](https://github.com/rphln/Mawile/commit/dc21d92da5e36035b70ebd89c152403d2d3371db))
* **DenseQPlayer:** Change the reward weights. ([6a341e8](https://github.com/rphln/Mawile/commit/6a341e88c0f374f49106424cb6119cf87335b613))
* **Train:** Always save snapshots of the model. ([908f51b](https://github.com/rphln/Mawile/commit/908f51beada0ab1d1b372cadb747a3d3c674bd5c))
* **Train:** Catch all exceptions when trying to load a model. ([12395cf](https://github.com/rphln/Mawile/commit/12395cfdcd30648956f80bc7a4d5400892fb02b4))
* **Train:** Fit the model for 10 epochs. ([82060c5](https://github.com/rphln/Mawile/commit/82060c5463f630b531123b30ec23f3d6e9e2f8ea))
* **Train:** Update the players' configuration. ([c0eb580](https://github.com/rphln/Mawile/commit/c0eb580ece0ca97ba7a609bde20ec60409325890))
