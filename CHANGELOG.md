# Changelog

## [0.3.0](https://github.com/constructorfleet/wyoming-chatterbox/compare/v0.2.0...v0.3.0) (2026-09-17)


### Features

* **obs:** add metrics instrumentation ([e91ad8f](https://github.com/constructorfleet/wyoming-chatterbox/commit/e91ad8f3063b86eac03b55386d699dbb9c87f953))
* Reduce turbo/nano voice prompt latency and add synthesis observability ([ebcb023](https://github.com/constructorfleet/wyoming-chatterbox/commit/ebcb02384eebb10bb9b4c0963163b8f4aff469d9))


### Bug Fixes

* **models:** key prompt cache by settings ([cf4f065](https://github.com/constructorfleet/wyoming-chatterbox/commit/cf4f065962d9866786c85983960ab4ea093f5c55))
* **models:** preserve empty prompt passthrough ([776700d](https://github.com/constructorfleet/wyoming-chatterbox/commit/776700dd72764bc830e9a8c8e202d271e643c9e1))
* **models:** preserve prompt passthrough ([23ea949](https://github.com/constructorfleet/wyoming-chatterbox/commit/23ea94930746f20ee4aabe3e0e2c40f46a90541d))
* **obs:** avoid shared prompt hook races ([c1a8680](https://github.com/constructorfleet/wyoming-chatterbox/commit/c1a86803c7d51441ace6d9ef9a8ff1f51290858d))
* **obs:** harden metrics startup and error stats ([6908869](https://github.com/constructorfleet/wyoming-chatterbox/commit/6908869b8136eaac8817997ab8fb72b17d65af1c))
* **obs:** tighten prompt cache safety ([c4fd062](https://github.com/constructorfleet/wyoming-chatterbox/commit/c4fd062421400991ce86e8dcf99bc7961723bbac))
* **server:** delay audio start until data ([fccada1](https://github.com/constructorfleet/wyoming-chatterbox/commit/fccada14aecb7e7c2a9fb144bd74bfab09946742))
* **server:** finalize default voice warmup handling ([7263f8d](https://github.com/constructorfleet/wyoming-chatterbox/commit/7263f8d00fbf77b72ab10c406180f6f4d7f5a499))
* **server:** make voice warmup best effort ([2651b9e](https://github.com/constructorfleet/wyoming-chatterbox/commit/2651b9e519c6e0f368ba9223c015534260b542f4))
* **server:** skip audio stop without start ([97d8c6c](https://github.com/constructorfleet/wyoming-chatterbox/commit/97d8c6cac466edf402694ecabf82fdaacc86c77f))


### Performance Improvements

* **voice:** cache and warm reference prompts ([a6b12d0](https://github.com/constructorfleet/wyoming-chatterbox/commit/a6b12d08d436c973a1507d117a2aec8f48c867e2))

## [0.2.0](https://github.com/constructorfleet/wyoming-chatterbox/compare/v0.1.3...v0.2.0) (2026-08-26)


### Features

* add conventional commit enforcement ([0bc9682](https://github.com/constructorfleet/wyoming-chatterbox/commit/0bc9682f70f02a58f5a475fd0b57b8c87f24a812))
* add conventional commit enforcement ([fab2346](https://github.com/constructorfleet/wyoming-chatterbox/commit/fab234637675c4d3df8643ed91d9c8510ae88862))

## [0.1.3](https://github.com/constructorfleet/wyoming-chatterbox/compare/v0.1.2...v0.1.3) (2026-08-26)


### Bug Fixes

* Install chatterbox from github and fix invocation ([c62bc4c](https://github.com/constructorfleet/wyoming-chatterbox/commit/c62bc4c19703aa8a4e54b17db1d952cb10911fac))
* Install chatterbox from github and fix invocation ([841f1dc](https://github.com/constructorfleet/wyoming-chatterbox/commit/841f1dc2914356208a0e68a392f61968fe329359))
* install git in Docker stages for chatterbox-tts git install ([0fbbed1](https://github.com/constructorfleet/wyoming-chatterbox/commit/0fbbed1ed412c2c59eabe742a5dc70bed2164924))
