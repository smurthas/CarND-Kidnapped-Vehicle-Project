/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of x, y, theta and their uncertainties from GPS)
  // and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  num_particles = 200;

  //weights = std::vector<double>(num_particles, 1.0);

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  default_random_engine gen;
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;
  std::normal_distribution<> dist_x(0.0, std_pos[0]);
  std::normal_distribution<> dist_y(0.0, std_pos[1]);
  std::normal_distribution<> dist_theta(0.0, std_pos[3]);

  // don't divide by zero
  if (abs(yaw_rate) < 0.00001) {
    yaw_rate = 0.00001;
  }

  for (Particle& p : particles) {
    double theta_1 = p.theta + yaw_rate*delta_t;
    p.x = p.x + velocity/yaw_rate*(sin(theta_1) - sin(p.theta)) + dist_x(gen);
    p.y = p.y + velocity/yaw_rate*(cos(p.theta) - cos(theta_1)) + dist_y(gen);
    p.theta = theta_1 + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to implement this method and use it as a helper
  // during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution

  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located according to the MAP'S coordinate system. You will
  // need to transform between the two systems. Keep in mind that this
  // transformation requires both rotation AND translation (but no scaling).

  // The following is a good resource for the theory:
  // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  // and the following is a good resource for the actual equation to implement
  // (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html

  double o_x = std_landmark[0];
  double o_y = std_landmark[1];

  double o_x_2_2 = 2.0 * pow(o_x, 2);
  double o_y_2_2 = 2.0 * pow(o_y, 2);
  double pi_std = 1.0 / (2.0 * M_PI * o_x * o_y);

  double weight_sum = 0;
  for (Particle& p : particles) {
    for (LandmarkObs obs : observations) {
      // transform observation to map coords
      LandmarkObs obs_t;
      obs_t.x = obs.x * cos(p.theta) - obs.y*sin(p.theta) + p.x;
      obs_t.y = obs.x * sin(p.theta) + obs.y*cos(p.theta) + p.y;

      // calculate to probability of this observation as the sum of all the
      // probabilities of this observation for each landmark
      // i.e. not just nearest neighbor, but "every neighbor"
      double p_obs = 0.0;
      for (Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
        float mu_x = landmark.x_f;
        float mu_y = landmark.y_f;
        p_obs +=  pi_std * exp(-1.0 * (pow(obs_t.x - mu_x, 2)/o_x_2_2 +
                                       pow(obs_t.y - mu_y, 2)/o_y_2_2));
      }

      p.weight *= p_obs;
    }

    weight_sum += p.weight;
  }


  // normalize weights
  for (Particle& p : particles) {
    p.weight /= weight_sum;
  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::default_random_engine generator;
  std::vector<double> weights;
  for (const Particle& p : particles) {
    weights.push_back(p.weight);
  }
  discrete_distribution<int> distr(weights.begin(), weights.end());

  std::vector<Particle> nextParticles;
  while (nextParticles.size() < particles.size()) {
    nextParticles.push_back(particles[distr(generator)]);
  }

  particles = nextParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's
  //          (x,y) world coordinates mapping to
  //   associations: The landmark id that goes along with each listed association
  //   sense_x: the associations x mapping already converted to world coordinates
  //   sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
