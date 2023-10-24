// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ecc

import (
	"crypto/elliptic"
	"crypto/sha256"
	"fmt"
	"math/big"
)

// HashToCurve 将字节数组哈希到指定的椭圆曲线上的点
// 警告：
// HashToCurve函数会一直重复执行哈希操作，直到找到一个在曲线上的点为止。
// 每次重试时，它会使用上一次的哈希结果作为输入。注意，这种方法不能保证在有限的时间内找到一个点。
//
// The P-256 curve is defined over the prime field GF(p), where p is:
// p = 2^256 - 2^224 + 2^192 + 2^96 - 1
//
// The P-384 curve is defined over the prime field GF(p), where p is:
// p = 2^384 - 2^128 - 2^96 + 2^32 - 1
func HashToCurve(msg []byte, curve elliptic.Curve) (*Point, error) {
	p := curve.Params().P

	timesTried := 1
	for {
		// 计算哈希值 h = SHA-256(msg)
		h := sha256.Sum256(msg)

		// 将哈希值转换为 big.Int
		x := new(big.Int).SetBytes(h[:])

		// 将 x 限制在曲线定义域内
		x.Mod(x, p)

		//		// 检查 x 是否对应于曲线上的一个 x 值
		//		if !isOnCurve(x, curve) {
		//			// 如果不是，重复执行哈希操作，new_msg = h = SHA-256(last_msg)
		//			msg = h[:]
		//			continue
		//		}

		// 计算 y^2 = x^3 + ax + b，其中 a, b 是曲线参数
		ySquared := new(big.Int).Exp(x, big.NewInt(3), p)

		// 根据曲线类别，来判断下参数a的数值
		if curve.Params().Name != "P-256" && curve.Params().Name != "SM2-P-256" {
			err := fmt.Errorf("curve [%v] is not supported yet.", curve.Params().Name)
			return nil, err
		}

		// NIST系列曲线的参数A是-3，先默认是NIST P-256曲线
		A := big.NewInt(-3)
		// SM2曲线的参数a
		if curve.Params().Name == "SM2-P-256" {
			A, _ = new(big.Int).SetString("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", 16)
		}

		// Compute ax
		//ax := new(big.Int).Mul(curve.Params().A, x)
		ax := new(big.Int).Mul(A, x)

		ySquared.Add(ySquared, ax)
		ySquared.Add(ySquared, curve.Params().B)

		// 计算 y 的平方根
		// 如果y是nil，就说明没找到合适的点
		//
		// 注意：NIST P-256曲线和国密SM2系列曲线的P满足p mod 4 = 3，所以可以使用费马小定理Fermat's Little Theorem 来快速计算平方根
		// Fermat's Little Theorem states that if p is a prime number and x is an integer not divisible by p,
		// then x^(p-1) is congruent to 1 modulo p:
		// x^(p-1) = 1 mod p
		// 因而：
		// 两边都乘以x：x^(p-1) * x^1 = x^(p-1+1) = x^p = x mod p
		// 两边再乘以x：x^(p-1) * x^2 = x^(p-1+2) = x^(p+1) = x^2 mod p
		// 因为p mod 4 = 3，所以p+1 = 4k，也就是说，可以被4整除
		// 两边的指数同时除以4：x^((p+1)/4) = x^(2/4) mod p = x^(1/2) mod p
		// 结论：
		// 假设有限域的范围是p，p为质数且 p mod 4 = 3, 根据费马小定理 a^(p-1) = 1 mod p,
		// 那么对于任意有限域的元素x，其在有限域的平方根y=x^(p+1)/4 mod p
		y := new(big.Int).ModSqrt(ySquared, p)

		// 如果 y^2 = x^3 + ax + b，则返回点 (x, y)
		if y != nil {
			//log.Printf("timesTried when HashToCurve: %d", timesTried)

			//			// Compute y = x^((p+1)/4) mod p
			//			log.Printf("equation p mod 4 is: %s", new(big.Int).Mod(elliptic.P256().Params().P, big.NewInt(4)))
			//
			//			exp := new(big.Int).Add(elliptic.P256().Params().P, big.NewInt(1))
			//			exp.Div(exp, big.NewInt(4))
			//			yTest := new(big.Int).Exp(ySquared, exp, elliptic.P256().Params().P)
			//
			//			// Check if y is really the square root of x
			//			y2 := new(big.Int).Mul(yTest, yTest)
			//			y2.Mod(y2, elliptic.P256().Params().P)
			//			ySquaredTest := new(big.Int).Mod(y2, elliptic.P256().Params().P)
			//			if y2.Cmp(ySquaredTest) != 0 {
			//				log.Printf("equation y = x^((p+1)/4) mod p is not enough for ModSqrt computation: y[%d], yTest[%d]", y, yTest)
			//			}

			//return x, y
			return &Point{Curve: curve, X: x, Y: y}, nil
		}

		// 如果没有找到合适的点，重复执行哈希操作
		msg = h[:]

		timesTried++
	}
}
